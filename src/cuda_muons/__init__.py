"""GPU muon transport: the CUDA kernel wrappers, the scattering tables and the fitted beam models."""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GMM_DIR = DATA_DIR / "gmm"
PARAMS_FILE = DATA_DIR / "tokanut_v6.txt"

_KERNEL = ("propagate_one_step", "sample_scatter", "set_environment")

__all__ = ["DATA_DIR", "GMM_DIR", "PARAMS_FILE", *_KERNEL]


def __getattr__(name):
    # Deferred so the data paths and the beam fitter stay importable without the built extension.
    if name in _KERNEL:
        from . import cuda_muons

        return getattr(cuda_muons, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
