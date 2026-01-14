# dataio.py
import os
import re
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# If you have physics.py in the same repo, these imports will work.
# compute_RR:     builds RR from A,B, wavelets, sin^2 list
# compute_relative_contrasts: returns dVp, dVs, dRho from Vp,Vs,Rho
try:
    from physics import compute_RR, compute_relative_contrasts
    _HAS_PHYS = True
except Exception:
    _HAS_PHYS = False

__all__ = [
    "RRDataset",
    "list_rr_indices",
    "create_folder_struct",
    "plot_loss",
    "write_summary",
    "setup_logging",
]

# ---------------------------
# Output / logging utilities
# ---------------------------

def create_folder_struct(base_dir: str = "../output") -> str:
    date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = os.path.join(base_dir, date)
    i = 1
    while os.path.exists(out_dir):
        out_dir = os.path.join(base_dir, f"{date}_{i}")
        i += 1
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output folder created: {out_dir}")
    return out_dir


def plot_loss(train_losses: Sequence[float], out_dir: str, fname: str = "loss_plot.png") -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print("Loss curve saved:", path)


def write_summary(model: torch.nn.Module, out_dir: str, fname: str = "model_summary.txt") -> None:
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        f.write(str(model))
        f.write("\n\nModel Parameters:\n")
        for name, p in model.named_parameters():
            f.write(f"{name}: {tuple(p.shape)}\n")


def setup_logging(
    out_dir: str,
    filename: str = "loss_record.log",
    level: int = logging.INFO,
) -> None:
    log_file = os.path.join(out_dir, filename)
    logging.basicConfig(
        level=level,
        filename=log_file,
        filemode="a",
        format="%(asctime)s   %(levelname)s   %(message)s",
    )
    logging.info("--- Logging initialized ---")

# ---------------------------
# Data utilities
# ---------------------------

_RX = re.compile(r"^RR_(\d{3})_angle(\d)\.npy$")


def list_rr_indices(data_dir: str) -> List[int]:
    indices = set()
    for fname in os.listdir(data_dir):
        if fname.startswith("RR_") and fname.endswith(".npy"):
            parts = fname.split("_")
            if len(parts) >= 3:
                try:
                    indices.add(int(parts[1]))
                except ValueError:
                    pass
    return sorted(indices)


def _compute_rr0_phys(
    *,
    bg_dir: str,
    wavelet_dir: Optional[str],
    sin2: Sequence[float],
    flip_z: bool = True,
    zero_mean_wavelets: bool = True,
    rr0_flip_after: bool = False,
) -> np.ndarray:
    """
    Build RR0_phys (4, H, W) from background models + wavelets, consistent with physics.py.
    """
    if not _HAS_PHYS:
        raise RuntimeError("physics.compute_RR / compute_relative_contrasts not available.")

    # Load backgrounds (float32)
    Por = np.load(os.path.join(bg_dir, "Porosity.npy")).astype(np.float32)
    Vp  = np.load(os.path.join(bg_dir, "Vp_syn.npy")).astype(np.float32)
    Vs  = np.load(os.path.join(bg_dir, "Vs_syn.npy")).astype(np.float32)
    Rho = np.load(os.path.join(bg_dir, "Rho_syn.npy")).astype(np.float32)

    # Optional vertical flip to match training orientation
    if flip_z:
        Por = Por[::-1].copy()
        Vp  = Vp[::-1].copy()
        Vs  = Vs[::-1].copy()
        Rho = Rho[::-1].copy()

    # Wavelets (4, kw)
    wdir = wavelet_dir or bg_dir
    wavelets = []
    for i in range(4):
        w = np.load(os.path.join(wdir, f"wavelet_{i}.npy")).astype(np.float32)
        if zero_mean_wavelets:
            w = w - w.mean()
        wavelets.append(torch.tensor(w))

    # Torch tensors for physics
    device = torch.device("cpu")
    Por_t = torch.tensor(Por, device=device)
    Vp_t  = torch.tensor(Vp,  device=device)
    Vs_t  = torch.tensor(Vs,  device=device)
    Rho_t = torch.tensor(Rho, device=device)

    dVp0, dVs0, dRho0 = compute_relative_contrasts(Vp_t, Vs_t, Rho_t)
    ratio0 = (Vs_t**2) / (Vp_t**2 + 1e-8)
    A0 = 0.5 * (dVp0 + dRho0)
    B0 = dVp0 - 2.0 * ratio0 * (dRho0 + 2.0 * dVs0)

    RR0_t = compute_RR(A0, B0, list(sin2), wavelets)  # (4,H,W) torch
    if rr0_flip_after:
        RR0_t = torch.flip(RR0_t, dims=[1])  # optional, seldom needed

    return RR0_t.detach().cpu().numpy().astype(np.float32)


class RRDataset(Dataset):
    """
    Returns dRR = RR_idx - RR_baseline for angles 0..3 as float32 (4,H,W).

    By default RR_baseline is read from files `RR_000_angle{a}.npy`.
    If `recompute_rr0=True`, RR_baseline is computed as RR0_phys from background
    models + wavelets (using physics.py), ensuring consistency with inference.

    Parameters
    ----------
    data_dir : str
        Directory containing RR_*_angle*.npy files (raw data snapshots).
    indices : Iterable[int]
        Snapshot indices to load.
    angles : tuple(int)
        Angles to include (default (0,1,2,3)).
    por_path : Optional[str]
        Optional Porosity passthrough (unchanged from original API).
    recompute_rr0 : bool
        If True, ignore file RR_000_angle*.npy and compute RR0_phys.
    rr0_params : Optional[Dict]
        Required when recompute_rr0=True.
        Keys:
            bg_dir: str            # where Porosity.npy, Vp_syn.npy, Vs_syn.npy, Rho_syn.npy live
            wavelet_dir: Optional[str]  # if None, use bg_dir
            sin2: Sequence[float]  # e.g., [0.0, 0.1, 0.2, 0.3]
            flip_z: bool           # match training orientation
            zero_mean_wavelets: bool
            rr0_flip_after: bool   # rarely needed; default False
    cache_dir : Optional[str]
        If set and recompute_rr0=True, cache RR0_phys as `{cache_dir}/RR0_phys_angle{i}.npy`.
        (If present, it will be loaded instead of recomputed.)
    prefer_cached : bool
        If True and cache_dir contains RR0_phys, load it; otherwise recompute and overwrite.

    Notes
    -----
    - dRR is computed on the fly as (RR_idx - RR0_baseline), where RR0_baseline is either
      the raw dataset baseline (RR_000_* files) or the physics-based RR0_phys.
    - Shapes of all angles are validated.
    """
    def __init__(
        self,
        data_dir: str,
        indices: Iterable[int],
        angles: Tuple[int, ...] = (0, 1, 2, 3),
        por_path: Optional[str] = None,
        recompute_rr0: bool = False,
        rr0_params: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
        prefer_cached: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.indices = list(indices)
        self.angles = tuple(angles)

        # Load one snapshot to determine (H, W)
        sample_files = []
        for a in self.angles:
            p = os.path.join(self.data_dir, f"RR_000_angle{a}.npy")
            if not os.path.exists(p):
                # if recomputing RR0_phys, we still need a shape; fall back to idx[0]
                fallback = os.path.join(self.data_dir, f"RR_{self.indices[0]:03d}_angle{a}.npy")
                if not os.path.exists(fallback):
                    raise FileNotFoundError(f"Missing both baseline and fallback for angle {a}.")
                sample_files.append(fallback)
            else:
                sample_files.append(p)

        shapes = [np.load(p, mmap_mode="r").shape for p in sample_files]
        if any(s != shapes[0] for s in shapes):
            raise ValueError(f"Angle shapes differ: {shapes}")
        self._shape = shapes[0]  # (H, W)

        # Prepare baseline
        self._use_phys_rr0 = bool(recompute_rr0)
        self._RR0 = None  # np.ndarray (4,H,W)
        if self._use_phys_rr0:
            if rr0_params is None:
                raise ValueError("rr0_params is required when recompute_rr0=True.")
            if not _HAS_PHYS:
                raise RuntimeError("physics.py not available; cannot recompute RR0_phys.")

            # try cache
            cached_ok = False
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cache_paths = [os.path.join(cache_dir, f"RR0_phys_angle{i}.npy") for i in range(len(self.angles))]
                if prefer_cached and all(os.path.exists(p) for p in cache_paths):
                    rr0_list = [np.load(p).astype(np.float32) for p in cache_paths]
                    self._RR0 = np.stack(rr0_list, axis=0)
                    cached_ok = True

            if not cached_ok:
                rr0 = _compute_rr0_phys(
                    bg_dir=rr0_params["bg_dir"],
                    wavelet_dir=rr0_params.get("wavelet_dir"),
                    sin2=rr0_params["sin2"],
                    flip_z=bool(rr0_params.get("flip_z", True)),
                    zero_mean_wavelets=bool(rr0_params.get("zero_mean_wavelets", True)),
                    rr0_flip_after=bool(rr0_params.get("rr0_flip_after", False)),
                )
                if rr0.shape[1:] != self._shape:
                    raise ValueError(f"RR0_phys shape {rr0.shape[1:]} != data shape {self._shape}")
                self._RR0 = rr0.astype(np.float32)

                # write cache
                if cache_dir:
                    for i in range(self._RR0.shape[0]):
                        np.save(os.path.join(cache_dir, f"RR0_phys_angle{i}.npy"), self._RR0[i])

        else:
            # dataset-provided baseline RR_000_angle*.npy
            self._base = []
            for a in self.angles:
                p = os.path.join(self.data_dir, f"RR_000_angle{a}.npy")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing baseline file: {p}")
                self._base.append(np.load(p, mmap_mode="r"))

        # optional porosity passthrough
        self.por = None
        if por_path:
            if not os.path.exists(por_path):
                raise FileNotFoundError(f"Porosity file not found: {por_path}")
            self.por = np.load(por_path).astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        diffs = []
        for k, a in enumerate(self.angles):
            curp = os.path.join(self.data_dir, f"RR_{idx:03d}_angle{a}.npy")
            if not os.path.exists(curp):
                raise FileNotFoundError(f"Missing snapshot: {curp}")
            cur = np.load(curp, mmap_mode="r")
            if cur.shape != self._shape:
                raise ValueError(f"Shape mismatch for angle {a}: {cur.shape} != {self._shape} at {curp}")

            if self._use_phys_rr0:
                base = self._RR0[k]
            else:
                base = self._base[k]
            diffs.append(cur.astype(np.float32) - base.astype(np.float32))

        x = torch.from_numpy(np.stack(diffs, axis=0))  # (4,H,W)
        if self.por is not None:
            por = torch.from_numpy(self.por)
            return x, por
        return x

