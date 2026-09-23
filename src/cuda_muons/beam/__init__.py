"""Fits the GMM beam models from the Geant4 muon sample."""

import os
from pathlib import Path

from cuda_muons import GMM_DIR

FIGS_DIR = Path(os.environ.get("MUON_FIGS_DIR", "figs")).expanduser()

__all__ = ["FIGS_DIR", "GMM_DIR", "sample_path"]


def sample_path(name: str = "full_sample.h5") -> Path:
    """Locate a Geant4 sample under ``$MUON_SAMPLE_DIR``; they are too large to ship."""
    root = os.environ.get("MUON_SAMPLE_DIR")
    if not root:
        raise RuntimeError(
            "set MUON_SAMPLE_DIR to the directory holding the Geant4 samples, e.g. "
            "MUON_SAMPLE_DIR=/path/to/muon_data"
        )
    path = Path(root).expanduser() / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path
