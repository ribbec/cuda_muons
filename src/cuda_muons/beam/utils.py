import torch
import h5py
import numpy as np

def get_real_muons(file_path, num_samples=None):
    print(f"Loading data from {file_path}...")
    with h5py.File(file_path, 'r') as f:
        px = f['px'][:] if num_samples is None else f['px'][:num_samples]
        py = f['py'][:] if num_samples is None else f['py'][:num_samples]
        pz = f['pz'][:] if num_samples is None else f['pz'][:num_samples]
        x_ = f['x'][:] if num_samples is None else f['x'][:num_samples]
        y_ = f['y'][:] if num_samples is None else f['y'][:num_samples]
        z_ = f['z'][:] if num_samples is None else f['z'][:num_samples]
        pdg = f['pdg'][:] if num_samples is None else f['pdg'][:num_samples]
        weight = f['weight'][:] if num_samples is None else f['weight'][:num_samples]
        data = np.stack([px, py, pz, x_, y_, z_, pdg, weight], axis=1)
    return data

def calc_pt(px, py):
    return np.sqrt(px**2 + py**2)

def calc_phi(px, py):
    return np.arctan2(py, px)

def generate_px_py(pt):
    phi = np.random.uniform(0, 2 * np.pi, size=pt.shape[0])
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return px, py


def to_gmm_features(pz, pt):
    """Map physical (pz, pt) [GeV] to the GMM's log feature space.

    pz -> log(pz)    (pz is bounded well above 0 for forward muons)
    pt -> log1p(pt)  (pt = |p_T| >= 0 and piles up near 0; log1p is finite at 0)

    Fitting the GMM in this space makes the right-skewed, non-negative momenta roughly
    Gaussian and ensures the inverse (from_gmm_features) returns pz>0 and pt>=0, removing
    the ~15% negative-tail artifact of fitting Gaussians directly on raw (pz, pt).
    Assumes pz>0 and pt>=0 (true for physical muon data).
    """
    return np.log(pz), np.log1p(pt)


def from_gmm_features(pz_feat, pt_feat):
    """Inverse of to_gmm_features: log-space features -> physical (pz, pt) [GeV]."""
    pz = np.exp(pz_feat)
    pt = np.clip(np.expm1(pt_feat), 0.0, None)
    return pz, pt


def generate_charge(num_samples):
    # Randomly assign charge +1 or -1 with equal probability
    charges = np.random.choice([-1, 1], size=num_samples)
    return charges
def generate_pdg(num_samples):
    # Randomly assign PDG code for muons (13 for muon, -13 for anti-muon)
    pdg_codes = (-13) * generate_charge(num_samples)
    return pdg_codes

def generate_position(num_samples, SmearBeamRadius=5.0, sigma=1.6):
    #ring transformation
    gauss_x = np.random.normal(0, sigma, size=num_samples)
    gauss_y = np.random.normal(0, sigma, size=num_samples)
    uniform = np.random.uniform(0, 1, size=num_samples)
    _phi = uniform * 2 * np.pi
    x = SmearBeamRadius * np.cos(_phi) + gauss_x
    y = SmearBeamRadius * np.sin(_phi) + gauss_y
    x = x / 100
    y = y / 100
    #z = np.zeros(num_samples) 
    return x, y#, z


def train_val_split(X, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X) * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], X[val_idx]


def standardize_train_val(X_train, X_val):
    # Accumulate in float64: float32 .mean()/.std() over tens of millions of rows saturates
    # (once the running sum exceeds ~2**24 the small per-element adds round away), which
    # silently corrupts the standardization stats — and hence the whole fit — for large N.
    mean = X_train.mean(axis=0, keepdims=True, dtype=np.float64)
    std = X_train.std(axis=0, keepdims=True, dtype=np.float64) + 1e-12
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    return X_train_s, X_val_s, mean, std


if __name__ == "__main__":
    from . import sample_path

    file_path = sample_path()
    data = get_real_muons(file_path)
    print(data.shape)

