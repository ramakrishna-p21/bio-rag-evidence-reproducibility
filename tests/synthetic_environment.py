"""A synthetic approved environment lock and a matching live inventory.

Nothing here is a real environment: the torch build, CUDA version, driver and GPU name are
invented, and the lock hash is registered only by tests that patch the approval registry.
"""

from biorag_repro.generation.environment import GpuDevice
from biorag_repro.generation.environment_lock import EnvironmentInventory

TORCH = "2.15.0+cu999"
CUDA = "99.9"
DRIVER = "0"
GPU = "Synthetic GPU"

LOCK = f"""\
schema_version = "1"
status = "approved_lock"
python = "3.11.9"
python_implementation = "CPython"
system = "Linux"
machine = "x86_64"
torch_cuda = "{CUDA}"
unresolved = []

[gpu]
min_count = 1
name_contains = "Synthetic"
nvidia_driver = "{DRIVER}"

[packages]
torch = "{TORCH}"
""".encode()


def device() -> GpuDevice:
    return GpuDevice(index=0, name=GPU, memory_total_mib=24576, driver_version=DRIVER)


def inventory(**changes) -> EnvironmentInventory:
    values = {
        "python_version": "3.11.9",
        "python_implementation": "CPython",
        "system": "Linux",
        "machine": "x86_64",
        "packages": {"torch": TORCH},
        "gpu_status": "ok",
        "gpus": [device()],
        "torch_status": "ok",
        "torch_cuda_build": CUDA,
        "torch_cuda_available": True,
        "torch_device_names": [GPU],
        "package_problems": [],
    }
    return EnvironmentInventory(**(values | changes))
