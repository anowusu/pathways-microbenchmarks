/*
 * Dynamic LD_PRELOAD patch for OSS pathways_server binaries to register the
 * PJRT_Profiler_Extension (PLUGIN_Profiler_Api) with tsl::ProfilerSession.
 *
 * When pathways_server loads libtpu.so and resolves "GetPjrtApi" via dlsym(),
 * this library intercepts the lookup, inspects the PJRT_Api extension chain for
 * PJRT_Extension_Type_Profiler (1), and registers a tsl::profiler::ProfilerFactory
 * that forwards Start(), Stop(), and CollectData(XSpace*) to the TPU PJRT plugin's
 * PLUGIN_Profiler_Api.
 *
 * If the pathways_server binary already contains native PJRT_Profiler_Extension
 * registration (e.g., once cl/1010067962 is merged upstream), this library
 * detects it automatically and becomes a no-op.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- PJRT & Plugin Profiler C API ABI definitions ---- */

#define PJRT_EXTENSION_TYPE_PROFILER 1

typedef struct PJRT_Extension_Base {
  size_t struct_size;
  int type;
  struct PJRT_Extension_Base* next;
} PJRT_Extension_Base;

typedef struct PJRT_Api_Header {
  size_t struct_size;
   PJRT_Extension_Base* extension_start;
} PJRT_Api_Header;

typedef struct PLUGIN_Profiler PLUGIN_Profiler;
typedef struct PLUGIN_Profiler_Error PLUGIN_Profiler_Error;
typedef struct PLUGIN_Profiler_ConsumeResult PLUGIN_Profiler_ConsumeResult;

typedef struct {
  size_t struct_size;
  void* priv;
  PLUGIN_Profiler_Error* error;
} PLUGIN_Profiler_Error_Destroy_Args;

typedef struct {
  size_t struct_size;
  void* priv;
  const PLUGIN_Profiler_Error* error;
  const char* message;
  size_t message_size;
} PLUGIN_Profiler_Error_Message_Args;

typedef struct {
  size_t struct_size;
  void* priv;
  const PLUGIN_Profiler_Error* error;
  int code;
} PLUGIN_Profiler_Error_GetCode_Args;

typedef struct {
  size_t struct_size;
  const char* options;
  size_t options_size;
  PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Create_Args;

typedef struct {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Destroy_Args;

typedef struct {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Start_Args;

typedef struct {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
} PLUGIN_Profiler_Stop_Args;

typedef struct {
  size_t struct_size;
  PLUGIN_Profiler* profiler;
  uint8_t* buffer;
  size_t buffer_size_in_bytes;
} PLUGIN_Profiler_CollectData_Args;

typedef struct PLUGIN_Profiler_Api {
  size_t struct_size;
  void* priv;
  void (*error_destroy)(PLUGIN_Profiler_Error_Destroy_Args* args);
  void (*error_message)(PLUGIN_Profiler_Error_Message_Args* args);
  PLUGIN_Profiler_Error* (*error_get_code)(
      PLUGIN_Profiler_Error_GetCode_Args* args);
  PLUGIN_Profiler_Error* (*create)(PLUGIN_Profiler_Create_Args* args);
  PLUGIN_Profiler_Error* (*destroy)(PLUGIN_Profiler_Destroy_Args* args);
  PLUGIN_Profiler_Error* (*start)(PLUGIN_Profiler_Start_Args* args);
  PLUGIN_Profiler_Error* (*stop)(PLUGIN_Profiler_Stop_Args* args);
  PLUGIN_Profiler_Error* (*collect_data)(
      PLUGIN_Profiler_CollectData_Args* args);
  void* consume;
  void* consume_result_destroy;
  void* serialize;
} PLUGIN_Profiler_Api;

typedef struct PJRT_Profiler_Extension {
  PJRT_Extension_Base base;
  const PLUGIN_Profiler_Api* profiler_api;
  intptr_t traceme_context_id;
} PJRT_Profiler_Extension;

/* ---- Dynamically resolved pathways_server symbols ---- */

typedef void (*RegisterProfilerFactoryFn)(void* std_function_buf);
typedef void (*CreateProfilersFn)(void* out_vector, const void* profile_options);
typedef bool (*SerializeToStringFn)(const void* msg_lite, void* cxx_string);
typedef bool (*MergeFromImplFn)(const char* ptr, size_t size, void* msg_lite,
                                const void* tc_table, int flags);

static pthread_mutex_t g_patch_mu = PTHREAD_MUTEX_INITIALIZER;
static bool g_patch_registered = false;
static const PLUGIN_Profiler_Api* g_profiler_api = NULL;

static RegisterProfilerFactoryFn g_register_profiler_factory = NULL;
static CreateProfilersFn g_create_profilers = NULL;
static void*** g_factories_ptr = NULL;
static SerializeToStringFn g_serialize_to_string = NULL;
static MergeFromImplFn g_merge_from_impl = NULL;
static const uint8_t* g_lea_started = NULL;
static uint64_t g_fallback_std_fn_vtable = 0;
static const uint8_t* g_fallback_factory_target_fn = NULL;

/* ---- Custom ProfilerInterface wrapper ---- */

typedef struct {
  const void** vtable;
  const PLUGIN_Profiler_Api* api;
  PLUGIN_Profiler* profiler;
} PjrtProfilerWrapper;

static const void* g_wrapper_vtable_storage[9];
static const void** g_wrapper_vtable_ptr = NULL;

static void destroy_plugin_error(const PLUGIN_Profiler_Api* api,
                                 PLUGIN_Profiler_Error* err) {
  if (!err || !api) return;
  if (api->error_message) {
    PLUGIN_Profiler_Error_Message_Args margs;
    memset(&margs, 0, sizeof(margs));
    margs.struct_size = sizeof(margs);
    margs.error = err;
    api->error_message(&margs);
    if (margs.message && margs.message_size > 0) {
      fprintf(stderr, "[pjrt_profiler_patch] plugin profiler error: %.*s\n",
              (int)margs.message_size, margs.message);
    }
  }
  if (api->error_destroy) {
    PLUGIN_Profiler_Error_Destroy_Args dargs;
    memset(&dargs, 0, sizeof(dargs));
    dargs.struct_size = sizeof(dargs);
    dargs.error = err;
    api->error_destroy(&dargs);
  }
}

static void wrapper_d1(PjrtProfilerWrapper* self) {
  if (self->api && self->profiler && self->api->destroy) {
    PLUGIN_Profiler_Destroy_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = self->profiler;
    PLUGIN_Profiler_Error* err = self->api->destroy(&args);
    if (err) destroy_plugin_error(self->api, err);
    self->profiler = NULL;
  }
}

static void wrapper_d0(PjrtProfilerWrapper* self) {
  wrapper_d1(self);
  free(self);
}

/* In this Abseil ABI, absl::OkStatus() has inlined rep_ == 1ULL. */
static uint64_t* wrapper_start(uint64_t* out_status,
                               PjrtProfilerWrapper* self) {
  if (self->api && self->profiler && self->api->start) {
    PLUGIN_Profiler_Start_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = self->profiler;
    PLUGIN_Profiler_Error* err = self->api->start(&args);
    if (err) destroy_plugin_error(self->api, err);
  }
  *out_status = 1ULL;
  return out_status;
}

static uint64_t* wrapper_stop(uint64_t* out_status, PjrtProfilerWrapper* self) {
  if (self->api && self->profiler && self->api->stop) {
    PLUGIN_Profiler_Stop_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = self->profiler;
    PLUGIN_Profiler_Error* err = self->api->stop(&args);
    if (err) destroy_plugin_error(self->api, err);
  }
  *out_status = 1ULL;
  return out_status;
}

/*
 * Computes the exact byte length of valid top-level protobuf wire fields in
 * `buf[0..len)`. Stops before any trailing 0x00 padding byte (such as the
 * `profiler_data_size + 1` buffer returned by PLUGIN_Profiler_CollectData).
 */
static size_t compute_valid_proto_wire_len(const uint8_t* buf, size_t len) {
  size_t pos = 0;
  while (pos < len) {
    size_t field_start = pos;
    uint64_t tag = 0;
    int shift = 0;
    while (pos < len) {
      uint8_t b = buf[pos++];
      tag |= (uint64_t)(b & 0x7f) << shift;
      if ((b & 0x80) == 0) break;
      shift += 7;
      if (shift >= 64) return field_start;
    }
    if (tag == 0 || (tag >> 3) == 0) {
      return field_start;
    }
    uint32_t wire_type = (uint32_t)(tag & 0x7);
    if (wire_type == 0) {
      while (pos < len && (buf[pos++] & 0x80) != 0) {
      }
    } else if (wire_type == 1) {
      if (pos + 8 > len) return field_start;
      pos += 8;
    } else if (wire_type == 2) {
      uint64_t slen = 0;
      int sshift = 0;
      while (pos < len) {
        uint8_t b = buf[pos++];
        slen |= (uint64_t)(b & 0x7f) << sshift;
        if ((b & 0x80) == 0) break;
        sshift += 7;
        if (sshift >= 64) return field_start;
      }
      if (pos + slen > len) return field_start;
      pos += (size_t)slen;
    } else if (wire_type == 5) {
      if (pos + 4 > len) return field_start;
      pos += 4;
    } else {
      return field_start;
    }
  }
  return pos;
}

static void merge_bytes_into_xspace(void* space, const uint8_t* buf,
                                    size_t len) {
  if (!space || !buf || len == 0 || !g_merge_from_impl) return;
  size_t valid_len = compute_valid_proto_wire_len(buf, len);
  if (valid_len == 0) return;

  void* vtable = *(void**)space;
  void* (*get_class_data)(void*) = *(void**)((char*)vtable + 0x20);
  void* class_data = get_class_data(space);
  const void* tc_table = *(const void**)((char*)class_data + 0x08);
  if (!tc_table) {
    void* desc_methods = *(void**)((char*)class_data + 0x40);
    const void* (*get_tc)(void*) = *(const void**)((char*)desc_methods + 0x10);
    tc_table = get_tc(space);
  }
  g_merge_from_impl((const char*)buf, valid_len, space, tc_table, 1);
}

static uint64_t* wrapper_collect_data(uint64_t* out_status,
                                      PjrtProfilerWrapper* self, void* space) {
  if (self->api && self->profiler && self->api->collect_data && space) {
    PLUGIN_Profiler_CollectData_Args args;
    memset(&args, 0, sizeof(args));
    args.struct_size = sizeof(args);
    args.profiler = self->profiler;
    args.buffer = NULL;
    args.buffer_size_in_bytes = 0;

    PLUGIN_Profiler_Error* err = self->api->collect_data(&args);
    if (err) {
      destroy_plugin_error(self->api, err);
    } else if (args.buffer != NULL && args.buffer_size_in_bytes > 0) {
      /*
       * Mode 1 (xla::profiler::PLUGIN_Profiler_CollectData in libtpu.so):
       * First call with args.buffer == NULL already populates args.buffer and
       * args.buffer_size_in_bytes.
       */
      merge_bytes_into_xspace(space, args.buffer, args.buffer_size_in_bytes);
    } else if (args.buffer == NULL && args.buffer_size_in_bytes > 0) {
      /*
       * Mode 2 (two-call C API contract): first call sets buffer_size_in_bytes,
       * caller allocates buffer and calls collect_data a second time.
       */
      uint8_t* heap_buf = (uint8_t*)calloc(1, args.buffer_size_in_bytes);
      if (heap_buf) {
        args.buffer = heap_buf;
        PLUGIN_Profiler_Error* err2 = self->api->collect_data(&args);
        if (err2) {
          destroy_plugin_error(self->api, err2);
        } else {
          merge_bytes_into_xspace(space, heap_buf, args.buffer_size_in_bytes);
        }
        free(heap_buf);
      }
    }
  }
  *out_status = 1ULL;
  return out_status;
}

static PjrtProfilerWrapper** create_pjrt_plugin_tracer(
    PjrtProfilerWrapper** out_unique_ptr, const void* profile_options) {
  *out_unique_ptr = NULL;
  if (!g_profiler_api || !g_profiler_api->create) {
    return out_unique_ptr;
  }

  /* Serialize tensorflow::ProfileOptions using pathways_server's MessageLite. */
  uint64_t cxx_str[3] = {0, 0, 0};
  const char* opt_ptr = "";
  size_t opt_len = 0;
  bool is_long = false;
  if (g_serialize_to_string && profile_options) {
    g_serialize_to_string(profile_options, cxx_str);
    if ((cxx_str[0] & 1ULL) == 0) {
      opt_ptr = ((const char*)cxx_str) + 1;
      opt_len = (size_t)(((uint8_t)cxx_str[0]) >> 1);
    } else {
      is_long = true;
      opt_len = (size_t)cxx_str[1];
      opt_ptr = (const char*)cxx_str[2];
    }
  }

  PLUGIN_Profiler_Create_Args args;
  memset(&args, 0, sizeof(args));
  args.struct_size = sizeof(args);
  args.options = opt_ptr;
  args.options_size = opt_len;
  args.profiler = NULL;

  PLUGIN_Profiler_Error* err = g_profiler_api->create(&args);
  if (is_long && cxx_str[2]) {
    free((void*)cxx_str[2]);
  }
  if (err) {
    destroy_plugin_error(g_profiler_api, err);
    return out_unique_ptr;
  }
  if (!args.profiler) {
    return out_unique_ptr;
  }

  PjrtProfilerWrapper* w =
      (PjrtProfilerWrapper*)calloc(1, sizeof(PjrtProfilerWrapper));
  w->vtable = g_wrapper_vtable_ptr;
  w->api = g_profiler_api;
  w->profiler = args.profiler;
  *out_unique_ptr = w;
  return out_unique_ptr;
}

/* ---- Dynamic binary pattern resolver (zero hardcoded offsets) ---- */

typedef struct {
  const uint8_t* ro_start;
  size_t ro_len;
  const uint8_t* rx_start;
  size_t rx_len;
  bool has_native_pjrt_profiler;
} MainBinarySegments;

static const uint8_t* find_bytes(const uint8_t* hay, size_t hay_len,
                                 const void* needle, size_t needle_len) {
  if (needle_len == 0 || hay_len < needle_len) return NULL;
  return (const uint8_t*)memmem(hay, hay_len, needle, needle_len);
}

static const uint8_t* find_lea_rsi_to_target(const uint8_t* rx_start,
                                             size_t rx_len,
                                             const uint8_t* target) {
  if (rx_len < 7) return NULL;
  for (size_t i = 0; i + 7 <= rx_len; ++i) {
    if (rx_start[i] == 0x48 && rx_start[i + 1] == 0x8d &&
        rx_start[i + 2] == 0x35) {
      int32_t disp;
      memcpy(&disp, &rx_start[i + 3], sizeof(disp));
      if (&rx_start[i + 7] + disp == target) {
        return &rx_start[i];
      }
    }
  }
  return NULL;
}

static const uint8_t* nth_call_target(const uint8_t* code, size_t max_scan,
                                      int n) {
  int count = 0;
  for (size_t i = 0; i + 5 <= max_scan; ++i) {
    if (code[i] == 0xe8) {
      count++;
      if (count == n) {
        int32_t disp;
        memcpy(&disp, &code[i + 1], sizeof(disp));
        return &code[i + 5] + disp;
      }
      i += 4;
    }
  }
  return NULL;
}

static int phdr_callback(struct dl_phdr_info* info, size_t size, void* data) {
  (void)size;
  MainBinarySegments* segs = (MainBinarySegments*)data;
  if (info->dlpi_name && info->dlpi_name[0] != '\0') {
    return 0;
  }
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    const ElfW(Phdr)* p = &info->dlpi_phdr[i];
    if (p->p_type != PT_LOAD) continue;
    const uint8_t* start = (const uint8_t*)(info->dlpi_addr + p->p_vaddr);
    size_t len = (size_t)p->p_memsz;
    if ((p->p_flags & PF_X) != 0) {
      segs->rx_start = start;
      segs->rx_len = len;
    } else if ((p->p_flags & PF_W) == 0 && (p->p_flags & PF_R) != 0) {
      segs->ro_start = start;
      segs->ro_len = len;
    }
  }
  return 1;
}

static bool resolve_pathways_symbols(void) {
  if (g_register_profiler_factory != NULL) return true;

  MainBinarySegments segs;
  memset(&segs, 0, sizeof(segs));
  dl_iterate_phdr(phdr_callback, &segs);
  if (!segs.ro_start || !segs.rx_start) {
    fprintf(stderr,
            "[pjrt_profiler_patch] Failed to locate PT_LOAD segments.\n");
    return false;
  }

  /* 1. Check if native PJRT_Profiler_Extension registration is already present. */
  const char* kNativeMarker = "PJRT_Profiler_Extension is not registered";
  if (find_bytes(segs.ro_start, segs.ro_len, kNativeMarker,
                 strlen(kNativeMarker)) != NULL) {
    fprintf(stderr,
            "[pjrt_profiler_patch] Binary already has native "
            "PJRT_Profiler_Extension registration; patch is a no-op.\n");
    g_patch_registered = true;
    return false;
  }

  /* 2. Locate ProfilerSession::ProfilerSession and CreateProfilers. */
  const char* kSessionStarted = "Profiler session started.";
  const uint8_t* str_started = find_bytes(
      segs.ro_start, segs.ro_len, kSessionStarted, strlen(kSessionStarted) + 1);
  if (!str_started) {
    fprintf(stderr, "[pjrt_profiler_patch] String '%s' not found.\n",
            kSessionStarted);
    return false;
  }
  const uint8_t* lea_started =
      find_lea_rsi_to_target(segs.rx_start, segs.rx_len, str_started);
  if (!lea_started) {
    fprintf(stderr, "[pjrt_profiler_patch] xref to '%s' not found.\n",
            kSessionStarted);
    return false;
  }
  g_lea_started = lea_started;

  const uint8_t* create_profilers = nth_call_target(lea_started, 0x80, 5);
  if (!create_profilers) {
    fprintf(stderr, "[pjrt_profiler_patch] CreateProfilers call not found.\n");
    return false;
  }
  g_create_profilers = (CreateProfilersFn)create_profilers;

  /* Inside CreateProfilers, find g_mu (48 8d 3d) and g_factories_ptr (48 8b 05). */
  const uint8_t* g_mu = NULL;
  for (size_t i = 0; i + 7 <= 0x60; ++i) {
    if (!g_mu && create_profilers[i] == 0x48 &&
        create_profilers[i + 1] == 0x8d && create_profilers[i + 2] == 0x3d) {
      int32_t disp;
      memcpy(&disp, &create_profilers[i + 3], sizeof(disp));
      g_mu = &create_profilers[i + 7] + disp;
    }
    if (!g_factories_ptr && create_profilers[i] == 0x48 &&
        create_profilers[i + 1] == 0x8b && create_profilers[i + 2] == 0x05) {
      int32_t disp;
      memcpy(&disp, &create_profilers[i + 3], sizeof(disp));
      g_factories_ptr = (void***)(&create_profilers[i + 7] + disp);
    }
  }
  if (!g_mu || !g_factories_ptr) {
    fprintf(stderr,
            "[pjrt_profiler_patch] Failed to locate g_mu / g_factories_ptr.\n");
    return false;
  }

  /* Scan backward up to 0x200 bytes before CreateProfilers for RegisterProfilerFactory. */
  for (size_t back = 0x40; back <= 0x200; ++back) {
    const uint8_t* fn = create_profilers - back;
    if (fn[0] == 0x55 && fn[1] == 0x48 && fn[2] == 0x89 && fn[3] == 0xe5) {
      for (size_t j = 4; j + 7 <= 0x30; ++j) {
        if (fn[j] == 0x48 && fn[j + 1] == 0x8d && fn[j + 2] == 0x3d) {
          int32_t disp;
          memcpy(&disp, &fn[j + 3], sizeof(disp));
          if (&fn[j + 7] + disp == g_mu) {
            g_register_profiler_factory = (RegisterProfilerFactoryFn)fn;
            break;
          }
        }
      }
      if (g_register_profiler_factory) break;
    }
  }
  if (!g_register_profiler_factory) {
    fprintf(stderr,
            "[pjrt_profiler_patch] Failed to locate RegisterProfilerFactory.\n");
    return false;
  }

  /* Also locate a static caller of RegisterProfilerFactory in .text as a
   * fallback for std_fn_vtable and factory_target_fn in case GetFactories()
   * has not been populated yet. */
  const uint8_t* reg_fn_bytes = (const uint8_t*)g_register_profiler_factory;
  const uint8_t* search_from =
      (reg_fn_bytes > segs.rx_start + 0x20000) ? (reg_fn_bytes - 0x20000)
                                               : segs.rx_start;
  for (const uint8_t* p = search_from; p + 5 < reg_fn_bytes; ++p) {
    if (p[0] == 0xe8) {
      int32_t disp;
      memcpy(&disp, &p[1], sizeof(disp));
      if (&p[5] + disp == reg_fn_bytes && p >= segs.rx_start + 0x30) {
        const uint8_t* l1 = NULL;
        const uint8_t* l2 = NULL;
        for (const uint8_t* q = p - 0x28; q + 7 <= p; ++q) {
          if (q[0] == 0x48 && q[1] == 0x8d && q[2] == 0x05) {
            int32_t ldisp;
            memcpy(&ldisp, &q[3], sizeof(ldisp));
            if (!l1) {
              l1 = &q[7] + ldisp;
            } else {
              l2 = &q[7] + ldisp;
            }
          }
        }
        if (l1 && l2) {
          g_fallback_std_fn_vtable = (uint64_t)l1;
          g_fallback_factory_target_fn = l2;
          break;
        }
      }
    }
  }

  /* 3. Locate MessageLite::SerializeToString and MergeFromImpl. */
  const char* kCollectingXSpace = "Collecting XSpace to repository: ";
  const uint8_t* str_coll =
      find_bytes(segs.ro_start, segs.ro_len, kCollectingXSpace,
                 strlen(kCollectingXSpace) + 1);
  if (str_coll) {
    const uint8_t* lea_coll =
        find_lea_rsi_to_target(segs.rx_start, segs.rx_len, str_coll);
    if (lea_coll) {
      const uint8_t* write_bin_proto = nth_call_target(lea_coll, 0x80, 6);
      if (write_bin_proto) {
        const uint8_t* ser_to_str = nth_call_target(write_bin_proto, 0x50, 1);
        if (ser_to_str) {
          g_serialize_to_string = (SerializeToStringFn)ser_to_str;
          /* Scan backward up to 0x400 bytes for `mov $1, %r8d` (41 b8 01 00 00 00)
           * followed by `jmp MergeFromImpl` (e9 disp32). */
          const uint8_t* scan_start = ser_to_str - 0x400;
          for (const uint8_t* p = scan_start; p + 32 < ser_to_str; ++p) {
            if (p[0] == 0x41 && p[1] == 0xb8 && p[2] == 0x01 && p[3] == 0x00 &&
                p[4] == 0x00 && p[5] == 0x00) {
              for (size_t k = 6; k <= 24; ++k) {
                if (p[k] == 0xe9) {
                  int32_t disp;
                  memcpy(&disp, &p[k + 1], sizeof(disp));
                  g_merge_from_impl = (MergeFromImplFn)(&p[k + 5] + disp);
                  break;
                }
              }
              if (g_merge_from_impl) break;
            }
          }
        }
      }
    }
  }
  if (!g_serialize_to_string || !g_merge_from_impl) {
    fprintf(stderr,
            "[pjrt_profiler_patch] Failed to locate MessageLite methods.\n");
    return false;
  }

  return true;
}

static void maybe_register_pjrt_profiler(const PJRT_Api_Header* pjrt_api) {
  if (!pjrt_api) return;

  pthread_mutex_lock(&g_patch_mu);
  if (g_patch_registered) {
    pthread_mutex_unlock(&g_patch_mu);
    return;
  }

  const PJRT_Profiler_Extension* prof_ext = NULL;
  for (PJRT_Extension_Base* ext = pjrt_api->extension_start; ext != NULL;
       ext = ext->next) {
    if (ext->type == PJRT_EXTENSION_TYPE_PROFILER) {
      prof_ext = (const PJRT_Profiler_Extension*)ext;
      break;
    }
  }
  if (!prof_ext || !prof_ext->profiler_api) {
    pthread_mutex_unlock(&g_patch_mu);
    return;
  }

  if (!resolve_pathways_symbols()) {
    pthread_mutex_unlock(&g_patch_mu);
    return;
  }

  /* Extract existing std::function vtable and HostTracer vtable. */
  uint64_t std_fn_vtable = g_fallback_std_fn_vtable;
  const uint8_t* factory_target_fn = g_fallback_factory_target_fn;
  if (g_factories_ptr && *g_factories_ptr && (*g_factories_ptr)[0]) {
    uint64_t* first_factory = (uint64_t*)((*g_factories_ptr)[0]);
    std_fn_vtable = first_factory[0];
    factory_target_fn = (const uint8_t*)first_factory[1];
  }
  if (!std_fn_vtable || !factory_target_fn) {
    fprintf(stderr,
            "[pjrt_profiler_patch] Failed to resolve factory vtable.\n");
    pthread_mutex_unlock(&g_patch_mu);
    return;
  }

  const void** base_vtable = NULL;
  for (size_t i = 0; i + 7 <= 0x60; ++i) {
    if (factory_target_fn[i] == 0x48 && factory_target_fn[i + 1] == 0x8d &&
        (factory_target_fn[i + 2] == 0x0d ||
         factory_target_fn[i + 2] == 0x05)) {
      int32_t disp;
      memcpy(&disp, &factory_target_fn[i + 3], sizeof(disp));
      base_vtable = (const void**)(&factory_target_fn[i + 7] + disp);
      break;
    }
  }

  g_wrapper_vtable_storage[0] = base_vtable ? base_vtable[-2] : NULL;
  g_wrapper_vtable_storage[1] = base_vtable ? base_vtable[-1] : NULL;
  g_wrapper_vtable_storage[2] = (const void*)&wrapper_d1;
  g_wrapper_vtable_storage[3] = (const void*)&wrapper_d0;
  g_wrapper_vtable_storage[4] = (const void*)&wrapper_start;
  g_wrapper_vtable_storage[5] = (const void*)&wrapper_stop;
  g_wrapper_vtable_storage[6] = (const void*)&wrapper_collect_data;
  g_wrapper_vtable_storage[7] = base_vtable ? base_vtable[5] : NULL;
  g_wrapper_vtable_storage[8] = base_vtable ? base_vtable[6] : NULL;
  g_wrapper_vtable_ptr = &g_wrapper_vtable_storage[2];

  g_profiler_api = prof_ext->profiler_api;

  uint64_t fn_buf[6];
  memset(fn_buf, 0, sizeof(fn_buf));
  fn_buf[0] = std_fn_vtable;
  fn_buf[1] = (uint64_t)&create_pjrt_plugin_tracer;
  fn_buf[4] = (uint64_t)&fn_buf[0];

  g_register_profiler_factory(fn_buf);
  g_patch_registered = true;

  fprintf(stderr,
          "[pjrt_profiler_patch] Registered PJRT_Profiler_Extension "
          "(profiler_api=%p, traceme_context_id=%ld) with tsl::ProfilerSession.\n",
          (const void*)g_profiler_api, (long)prof_ext->traceme_context_id);
  pthread_mutex_unlock(&g_patch_mu);
}

/* ---- Intercept dlsym("GetPjrtApi") ---- */

typedef const PJRT_Api_Header* (*GetPjrtApiFn)(void);
static GetPjrtApiFn g_real_get_pjrt_api = NULL;

static const PJRT_Api_Header* wrapped_get_pjrt_api(void) {
  const PJRT_Api_Header* api =
      g_real_get_pjrt_api ? g_real_get_pjrt_api() : NULL;
  if (api) {
    maybe_register_pjrt_profiler(api);
  }
  return api;
}

typedef void* (*DlsymFn)(void*, const char*);
extern void* __libc_dlsym(void* handle, const char* name);

static DlsymFn get_real_dlsym(void) {
  static DlsymFn real_fn = NULL;
  if (!real_fn) {
    real_fn = (DlsymFn)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  }
  return real_fn;
}

void* dlsym(void* handle, const char* symbol) {
  DlsymFn real_dlsym = get_real_dlsym();
  void* res = real_dlsym ? real_dlsym(handle, symbol) : NULL;
  const char* volatile sym_ptr = symbol;
  if (res && sym_ptr && strcmp(sym_ptr, "GetPjrtApi") == 0) {
    g_real_get_pjrt_api = (GetPjrtApiFn)res;
    /* Also attempt eager registration if GetPjrtApi() returns a ready table. */
    const PJRT_Api_Header* api = g_real_get_pjrt_api();
    if (api) {
      maybe_register_pjrt_profiler(api);
    }
    return (void*)&wrapped_get_pjrt_api;
  }
  return res;
}

/* ---- Self-test harness (enabled when PJRT_PROFILER_PATCH_SELFTEST=1) ---- */

static int g_selftest_started = 0;
static int g_selftest_start_calls = 0;
static int g_selftest_stop_calls = 0;
static int g_selftest_collect_calls = 0;

/* Minimal serialized XSpace containing a single XPlane named "/device:TPU:0"
 * plus a trailing 0x00 byte to mimic libtpu.so's `profiler_data_size + 1`.
 * XSpace.planes (field 1, wire type 2 -> 0x0a, len 15) ->
 *   XPlane.name (field 1, wire type 2 -> 0x0a, len 13, "/device:TPU:0")
 */
static uint8_t kMockTpuXSpaceBytes[] = {
    0x0a, 0x0f, 0x0a, 0x0d, '/', 'd', 'e', 'v', 'i', 'c',
    'e',  ':',  'T',  'P',  'U', ':', '0', 0x00,
};

static PLUGIN_Profiler_Error* mock_create(PLUGIN_Profiler_Create_Args* args) {
  args->profiler = (PLUGIN_Profiler*)0x12345678ULL;
  return NULL;
}
static PLUGIN_Profiler_Error* mock_destroy(PLUGIN_Profiler_Destroy_Args* args) {
  (void)args;
  return NULL;
}
static PLUGIN_Profiler_Error* mock_start(PLUGIN_Profiler_Start_Args* args) {
  (void)args;
  g_selftest_start_calls++;
  return NULL;
}
static PLUGIN_Profiler_Error* mock_stop(PLUGIN_Profiler_Stop_Args* args) {
  (void)args;
  g_selftest_stop_calls++;
  return NULL;
}
static PLUGIN_Profiler_Error* mock_collect(
    PLUGIN_Profiler_CollectData_Args* args) {
  g_selftest_collect_calls++;
  args->buffer = kMockTpuXSpaceBytes;
  args->buffer_size_in_bytes = sizeof(kMockTpuXSpaceBytes);
  return NULL;
}

static PLUGIN_Profiler_Api g_mock_profiler_api = {
    sizeof(PLUGIN_Profiler_Api),
    NULL,
    NULL,
    NULL,
    NULL,
    mock_create,
    mock_destroy,
    mock_start,
    mock_stop,
    mock_collect,
    NULL,
    NULL,
    NULL,
};

static PJRT_Profiler_Extension g_mock_ext = {
    {sizeof(PJRT_Profiler_Extension), PJRT_EXTENSION_TYPE_PROFILER, NULL},
    &g_mock_profiler_api,
    42,
};

static PJRT_Api_Header g_mock_pjrt_api = {
    sizeof(PJRT_Api_Header),
    &g_mock_ext.base,
};

const PJRT_Api_Header* GetPjrtApi(void) { return &g_mock_pjrt_api; }

typedef int (*MainFn)(int, char**, char**);
static MainFn g_real_main = NULL;

static int selftest_main(int argc, char** argv, char** envp) {
  (void)argc;
  (void)argv;
  (void)envp;
  fprintf(stderr, "[pjrt_profiler_patch] Running self-test...\n");

  /* Trigger dlsym("GetPjrtApi") interception */
  GetPjrtApiFn fn = (GetPjrtApiFn)dlsym(RTLD_DEFAULT, "GetPjrtApi");
  if (!fn) {
    fprintf(stderr, "[pjrt_profiler_patch] SELFTEST FAIL: dlsym returned NULL\n");
    return 1;
  }
  const PJRT_Api_Header* api = fn();
  if (!api || !g_patch_registered) {
    fprintf(stderr,
            "[pjrt_profiler_patch] SELFTEST FAIL: patch not registered\n");
    return 1;
  }

  /* Verify GetFactories() now has 3 factories (Python, Host, PJRT) */
  uint64_t* vec = (uint64_t*)(*g_factories_ptr);
  size_t num_factories = (vec[1] - vec[0]) / 48;
  fprintf(stderr, "[pjrt_profiler_patch] Registered factories count: %zu\n",
          num_factories);
  if (num_factories < 3) {
    fprintf(stderr,
            "[pjrt_profiler_patch] SELFTEST FAIL: expected >= 3 factories\n");
    return 1;
  }

  /* Directly exercise our registered factory and wrapper methods */
  PjrtProfilerWrapper* wrapper = NULL;
  create_pjrt_plugin_tracer(&wrapper, NULL);
  if (!wrapper) {
    fprintf(stderr, "[pjrt_profiler_patch] SELFTEST FAIL: wrapper is NULL\n");
    return 1;
  }
  uint64_t st = 0;
  wrapper_start(&st, wrapper);
  wrapper_stop(&st, wrapper);

  /* Locate Arena::CreateMaybeMessage<XSpace>(NULL) right before
   * ProfilerSession::CollectData and test wrapper_collect_data + XSpace merge! */
  void* (*create_xspace)(void*) = NULL;
  if (g_lea_started) {
    for (size_t back = 0x100; back <= 0xc00; ++back) {
      const uint8_t* fn = g_lea_started - back;
      if (fn[0] == 0x55 && fn[1] == 0x48 && fn[2] == 0x89 && fn[3] == 0xe5 &&
          find_bytes(fn, 0x30, "\xbf\x78\x00\x00\x00", 5) != NULL) {
        create_xspace = (void* (*)(void*))fn;
        break;
      }
    }
  }
  if (!create_xspace) {
    fprintf(stderr,
            "[pjrt_profiler_patch] SELFTEST FAIL: create_xspace not found\n");
    return 1;
  }
  void* xspace = create_xspace(NULL);
  wrapper_collect_data(&st, wrapper, xspace);

  uint64_t out_str[3] = {0, 0, 0};
  g_serialize_to_string(xspace, out_str);
  const char* ser_ptr = (out_str[0] & 1ULL) == 0 ? (((const char*)out_str) + 1)
                                                 : (const char*)out_str[2];
  size_t ser_len = (out_str[0] & 1ULL) == 0 ? (size_t)(((uint8_t)out_str[0]) >> 1)
                                            : (size_t)out_str[1];
  if (ser_len != sizeof(kMockTpuXSpaceBytes) - 1 ||
      memcmp(ser_ptr, kMockTpuXSpaceBytes, ser_len) != 0) {
    fprintf(stderr,
            "[pjrt_profiler_patch] SELFTEST FAIL: XSpace serialized len=%zu "
            "(expected %zu)\n",
            ser_len, sizeof(kMockTpuXSpaceBytes) - 1);
    return 1;
  }
  fprintf(stderr,
          "[pjrt_profiler_patch] Verified merged XSpace contains '/device:TPU:0' "
          "(%zu bytes)!\n",
          ser_len);

  wrapper_d0(wrapper);

  if (g_selftest_start_calls != 1 || g_selftest_stop_calls != 1 ||
      g_selftest_collect_calls != 1) {
    fprintf(stderr,
            "[pjrt_profiler_patch] SELFTEST FAIL: start=%d stop=%d collect=%d\n",
            g_selftest_start_calls, g_selftest_stop_calls,
            g_selftest_collect_calls);
    return 1;
  }

  fprintf(stderr,
          "[pjrt_profiler_patch] SELFTEST PASS: all symbols resolved and "
          "PJRT profiler factory verified!\n");
  return 0;
}

int __libc_start_main(MainFn main_fn, int argc, char** ubp_av,
                      void (*init)(void), void (*fini)(void),
                      void (*rtld_fini)(void), void* stack_end) {
  typedef int (*LibcStartMainFn)(MainFn, int, char**, void (*)(void),
                                 void (*)(void), void (*)(void), void*);
  LibcStartMainFn real_start =
      (LibcStartMainFn)dlvsym(RTLD_NEXT, "__libc_start_main", "GLIBC_2.2.5");
  const char* selftest = getenv("PJRT_PROFILER_PATCH_SELFTEST");
  if (selftest && strcmp(selftest, "1") == 0 && !g_selftest_started) {
    g_selftest_started = 1;
    g_real_main = main_fn;
    return real_start(selftest_main, argc, ubp_av, init, fini, rtld_fini,
                      stack_end);
  }
  return real_start(main_fn, argc, ubp_av, init, fini, rtld_fini, stack_end);
}
