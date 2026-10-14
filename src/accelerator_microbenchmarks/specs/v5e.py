"""TPU v5e hardware specification."""

from accelerator_microbenchmarks.specs import schema

# See also spec in https://docs.cloud.google.com/tpu/docs/v5e and the
# per-chip comparison table in
# https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm.
#
# Values follow the same conventions as the v6e spec:
# * `float32` is 2-pass BF16 emulation on the MXU (half the BF16 peak).
# * ICI is the bidirectional ICI bandwidth per chip (v5e: 400 GB/s over 4 ICI
#   ports), stored in the same field as the other generations.
# * HBM is the per-chip HBM2 bandwidth from the datasheet (819 GB/s), using the
#   same 800 * 1.024 convention as v6e's 1600 * 1.024 = 1638.4.
V5E_HARDWARE_SPEC = schema.HardwareSpec(
    name=schema.TpuVersion.V5E,
    topology_dimension=2,
    devices_per_chip=1,
    tflops=schema.TflopsSpec(
        peak_tflops_per_device={
            "bfloat16": 197.0,
            "float32": 98.5,  # 2-pass BF16 emulation on MXU
            "int8": 394.0,
            "float8_e5m2": None,  # Not natively supported on v5e.
            "float8_e4m3fn": None,  # Not natively supported on v5e.
            "int4": None,  # Not natively supported on v5e.
            "float4_e2m1fn": None,  # Not natively supported on v5e.
        }
    ),
    ici=schema.IciSpec(
        unidirectional_link_bw_gb_s=400.0,
        bidirectional=True,
    ),
    hbm=schema.HbmSpec(peak_bw_gb_s=819.2),
)
