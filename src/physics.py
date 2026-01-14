# physics.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

EPS = 1e-8

# ------------------------ core rock/fluid physics ------------------------

def gassmann_substitution(Por: torch.Tensor,
                          Sw2: torch.Tensor,
                          RhoS: float, RhoW: float, RhoNw: float,
                          Vp1: torch.Tensor, Vs1: torch.Tensor):
    """
    Gassmann substitution with broadcasting.
    Inputs
      Por, Vp1, Vs1: (nz, nx)
      Sw2: (B, 1, nz, nx)   -- water saturation
    Returns
      Vp2, Vs2, Rho2: (B, 1, nz, nx)
    """
    device, dtype = Por.device, Por.dtype

    # constants on correct device/dtype
    RhoS = torch.as_tensor(RhoS, device=device, dtype=dtype)
    RhoW = torch.as_tensor(RhoW, device=device, dtype=dtype)
    RhoNw = torch.as_tensor(RhoNw, device=device, dtype=dtype)
    CompS  = torch.as_tensor(2.71e-11, device=device, dtype=dtype)
    CompW  = torch.as_tensor(3.6e-8,   device=device, dtype=dtype)
    CompNw = torch.as_tensor(6.3e-10,  device=device, dtype=dtype)

    K0 = 1.0 / (CompS + EPS)
    Sw1 = torch.ones_like(Por)

    fluid1 = Sw1 * RhoW + (1.0 - Sw1) * RhoNw
    Rho1 = (1.0 - Por) * RhoS + Por * fluid1

    K1   = Rho1 * (Vp1**2 - (4.0/3.0) * Vs1**2)
    KFl1 = 1.0 / (Sw1 * CompW + (1.0 - Sw1) * CompNw + EPS)

    RhoDry = Rho1 - Por * fluid1
    num1 = (K1 / (K0 - K1 + EPS)) - (KFl1 / (Por * (K0 - KFl1 + EPS) + EPS))
    den1 = 1.0 + (K1 / (K0 - K1 + EPS)) - (KFl1 / (Por * (K0 - KFl1 + EPS) + EPS))
    KDry = K0 * (num1 / (den1 + EPS))

    # target fluid mix (broadcast with Sw2)
    KFl2 = 1.0 / (Sw2 * CompW + (1.0 - Sw2) * CompNw + EPS)
    Rho2 = RhoDry + Por * (Sw2 * RhoW + (1.0 - Sw2) * RhoNw)

    num2 = (KDry / (K0 - KDry + EPS)) + (KFl2 / (Por * (K0 - KFl2 + EPS) + EPS))
    den2 = 1.0 + (KDry / (K0 - KDry + EPS)) + (KFl2 / (Por * (K0 - KFl2 + EPS) + EPS))
    K2   = K0 * (num2 / (den2 + EPS))

    # clamp for stability
    Rho2 = torch.clamp(Rho2, min=EPS)
    Vs2  = Vs1 * torch.sqrt(torch.clamp(Rho1 / Rho2, min=EPS))
    Vp2  = torch.sqrt(torch.clamp((K2 + (4.0/3.0) * Rho2 * Vs2**2) / (Rho2 + EPS), min=EPS))

    return Vp2, Vs2, Rho2  # (B,1,nz,nx)


# ------------------------ contrasts (batched) ------------------------
def compute_relative_contrasts(Vp, Vs, Rho):
    dVp  = Vp[1:, :]  - Vp[:-1, :]
    dVs  = Vs[1:, :]  - Vs[:-1, :]
    dRho = Rho[1:, :] - Rho[:-1, :]

    Vp_avg  = 0.5 * (Vp[1:, :]  + Vp[:-1, :])
    Vs_avg  = 0.5 * (Vs[1:, :]  + Vs[:-1, :])
    Rho_avg = 0.5 * (Rho[1:, :] + Rho[:-1, :])

    dVp_rel  = dVp  / (Vp_avg  + EPS)
    dVs_rel  = dVs  / (Vs_avg  + EPS)
    dRho_rel = dRho / (Rho_avg + EPS)

    # pad one row at the TOP by replicating the first valid row
    dVp_rel  = torch.cat([dVp_rel[:1, :],  dVp_rel],  dim=0)
    dVs_rel  = torch.cat([dVs_rel[:1, :],  dVs_rel],  dim=0)
    dRho_rel = torch.cat([dRho_rel[:1, :], dRho_rel], dim=0)
    return dVp_rel, dVs_rel, dRho_rel


def compute_relative_contrasts_batched(Vp, Vs, Rho):
    dVp  = Vp[:, 1:, :] - Vp[:, :-1, :]
    dVs  = Vs[:, 1:, :] - Vs[:, :-1, :]
    dRho = Rho[:, 1:, :] - Rho[:, :-1, :]

    Vp_avg  = 0.5 * (Vp[:, 1:, :]  + Vp[:, :-1, :])
    Vs_avg  = 0.5 * (Vs[:, 1:, :]  + Vs[:, :-1, :])
    Rho_avg = 0.5 * (Rho[:, 1:, :] + Rho[:, :-1, :])

    dVp_rel  = dVp  / (Vp_avg  + EPS)
    dVs_rel  = dVs  / (Vs_avg  + EPS)
    dRho_rel = dRho / (Rho_avg + EPS)

    # pad one row at the TOP for each batch item
    dVp_rel  = torch.cat([dVp_rel[:, :1, :],  dVp_rel],  dim=1)
    dVs_rel  = torch.cat([dVs_rel[:, :1, :],  dVs_rel],  dim=1)
    dRho_rel = torch.cat([dRho_rel[:, :1, :], dRho_rel], dim=1)
    return dVp_rel, dVs_rel, dRho_rel


# ------------------------ reflectivity (convolution along depth) ------------------------
def _prep_wavelet(w: torch.Tensor, device, dtype) -> torch.Tensor:
    """Ensure 1D, correct device/dtype, and zero-mean (to suppress DC)."""
    w = w.to(device=device, dtype=dtype).flatten()
    # If you want strict parity with some older runs, comment the next line:
    w = w - w.mean()
    return w

def compute_RR(A: torch.Tensor, B: torch.Tensor,
               sin2_list: List[float], wavelets: List[torch.Tensor]) -> torch.Tensor:
    """
    A, B: (nz, nx)
    Returns: (n_angles, nz, nx)
    Uses replicate padding along depth to avoid edge artifacts.
    """
    nz, nx = A.shape
    RR = torch.empty((len(sin2_list), nz, nx), device=A.device, dtype=A.dtype)

    for i, s2 in enumerate(sin2_list):
        R = A + B * s2                               # (nz, nx)
        w = _prep_wavelet(wavelets[i], A.device, A.dtype)
        kw = int(w.numel())
        assert kw % 2 == 1, "wavelet kernel must have odd length"
        pad = (kw - 1) // 2

        # per-trace conv along depth
        x = R.permute(1, 0).unsqueeze(1).contiguous()         # (nx, 1, nz)
        x = F.pad(x, (pad, pad), mode="replicate")            # replicate pad along depth
        y = F.conv1d(x, w.view(1, 1, kw), padding=0)          # (nx, 1, nz)
        RR[i] = y.squeeze(1).permute(1, 0).contiguous()       # (nz, nx)

    return RR


def compute_RR_batched(A: torch.Tensor, B: torch.Tensor,
                       sin2_list: List[float], wavelets: List[torch.Tensor]) -> torch.Tensor:
    """
    A, B: (B, nz, nx)
    Returns: (B, n_angles, nz, nx)
    Memory-friendly: reshape to (B*nx, 1, nz) and convolve once per angle.
    Uses replicate padding along depth to avoid edge artifacts.
    """
    Bsz, nz, nx = A.shape
    outs = []

    for i, s2 in enumerate(sin2_list):
        R = (A + B * s2).permute(0, 2, 1).contiguous()        # (B, nx, nz)
        w = _prep_wavelet(wavelets[i], A.device, A.dtype)
        kw = int(w.numel())
        assert kw % 2 == 1, "wavelet kernel must have odd length"
        pad = (kw - 1) // 2

        x = R.reshape(Bsz * nx, 1, nz)                        # (B*nx, 1, nz)
        x = F.pad(x, (pad, pad), mode="replicate")            # replicate pad along depth
        y = F.conv1d(x, w.view(1, 1, kw), padding=0)          # (B*nx, 1, nz)
        outs.append(y.view(Bsz, nx, nz).permute(0, 2, 1).contiguous())  # (B, nz, nx)

    return torch.stack(outs, dim=1)                           # (B, n_angles, nz, nx)

# ------------------------ full forward (batched) ------------------------

def compute_dRR_from_saturation_batch(
    Snw, Vp1, Vs1, Por, sin2, *, wavelets: List[torch.Tensor]
):
    """
    Fixed-wavelet forward model.
    Snw: (B,1,nz,nx) in [0,1]
    Vp1, Vs1, Por: (nz,nx)
    sin2: list of floats (length = n_angles)
    Returns: (B, n_angles, nz, nx)
    """
    B = Snw.shape[0]
    Sw2 = 1.0 - Snw

    RhoW, RhoNw, RhoS = 1000.0, 700.0, 2500.0
    Vp2, Vs2, Rho2 = gassmann_substitution(Por, Sw2, RhoS, RhoW, RhoNw, Vp1, Vs1)
    Vp2 = Vp2[:, 0]; Vs2 = Vs2[:, 0]; Rho2 = Rho2[:, 0]

    dVp, dVs, dRho = compute_relative_contrasts_batched(Vp2, Vs2, Rho2)
    ratio = (Vs2 ** 2) / (Vp2 ** 2 + EPS)
    A = 0.5 * (dVp + dRho)
    Bterm = dVp - 2.0 * ratio * (dRho + 2.0 * dVs)

    return compute_RR_batched(A, Bterm, sin2, wavelets)