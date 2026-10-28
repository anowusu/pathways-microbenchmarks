"""TPU v5e hardware specification."""

from accelerator_microbenchmarks.specs import schema

# See also spec in https://docs.cloud.google.com/tpu/docs/v5e
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
        link_bw_per_direction_gb_s=50.0,
        wraparound_rule=schema.WraparoundRule.PER_DIM_GE_16,
    ),
    hbm=schema.HbmSpec(peak_bw_gb_s=819.2),
)
