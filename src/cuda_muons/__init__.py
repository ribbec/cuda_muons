"""GPU muon transport: the CUDA kernel wrappers, the scattering tables and the fitted beam models."""

from pathlib import Path

from .cuda_muons import propagate_one_step, sample_scatter, set_environment

DATA_DIR = Path(__file__).parent / "data"
GMM_DIR = DATA_DIR / "gmm"
PARAMS_FILE = DATA_DIR / "tokanut_v6.txt"

__all__ = [
    "DATA_DIR",
    "GMM_DIR",
    "PARAMS_FILE",
    "propagate_one_step",
    "sample_scatter",
    "set_environment",
]
