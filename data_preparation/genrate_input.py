'''
import os
import numpy as np

# Load baseline models
Rho_model = np.flip(np.load('Rho_baseline.npy'), axis=0)
Vp_model = np.flip(np.load('Vp_baseline.npy'), axis=0)
Vs_model = np.flip(np.load('Vs_baseline.npy'), axis=0)

# Dimensions
nz_model, nx_model = Rho_model.shape

# Overburden
Vp_over = np.flip(np.load('Vp_overburden.npy'), axis=0)
Vs_over = np.flip(np.load('Vs_overburden.npy'), axis=0)
Rho_over = np.flip(np.load('Rho_overburden.npy'), axis=0)

# Porosity
PV = np.load('porous_volume.npy')
Por = PV / 250.0

# Saturation baseline
Snw = 0.0 * np.ones([nz_model, nx_model])  # initial CO2 saturation
Sw = 1 - Snw

# Rock and fluid properties
RhoS = 2500
CompS = 2.71e-11
K0 = 1 / CompS

RhoNw = 700
CompNw = 6.3e-10

RhoW = 1000
CompW = 3.6e-8

# Wavelets
wav_0 = np.load('wavelet_0.npy')
wav_1 = np.load('wavelet_1.npy')
wav_2 = np.load('wavelet_2.npy')
wav_3 = np.load('wavelet_3.npy')

wav = np.stack([wav_0, wav_1, wav_2, wav_3], axis=0)

# Angles
sin2 = [0.0, 0.1, 0.2, 0.3]

def substitution(Por, Snw2, RhoS, RhoW, RhoNw, Vp_S1, Vs_S1):
    Sw = 1 - Snw
    Sw2 = 1 - Snw2

    # State 1
    Rho_S1 = (1 - Por) * RhoS + Por * (Sw * RhoW + (1 - Sw) * RhoNw)
    K_S1 = Rho_S1 * (Vp_S1 ** 2 - (4 / 3) * Vs_S1 ** 2)
    CompFl = Sw * CompW + (1 - Sw) * CompNw
    KFl = 1 / CompFl

    # Dry Rock
    Rho_Dry = Rho_S1 - Por * (Sw * RhoW + (1 - Sw) * RhoNw)
    K_Dry = K0 * (K_S1 / (K0 - K_S1) - KFl / (Por * (K0 - KFl))) / (1 + (K_S1 / (K0 - K_S1) - KFl / (Por * (K0 - KFl))))
    Vp_Dry = np.sqrt((K_Dry + (4 / 3) * Rho_S1 * Vs_S1 ** 2) / Rho_Dry)
    Vs_Dry = Vs_S1 * np.sqrt(Rho_S1 / Rho_Dry)

    # State 2
    RhoNw2 = 700
    CompNw2 = 6.3e-10
    CompFl2 = Sw2 * CompW + (1 - Sw2) * CompNw2
    KFl2 = 1 / CompFl2

    Rho_S2 = Rho_Dry + Por * (Sw2 * RhoW + (1 - Sw2) * RhoNw2)
    K_S2 = K0 * (K_Dry / (K0 - K_Dry) + KFl2 / (Por * (K0 - KFl2))) / (1 + (K_Dry / (K0 - K_Dry) + KFl2 / (Por * (K0 - KFl2))))
    Vs_S2 = Vs_Dry * np.sqrt(Rho_Dry / Rho_S2)
    Vp_S2 = np.sqrt((K_S2 + (4 / 3) * Rho_S2 * Vs_S2 ** 2) / Rho_S2)

    return Vp_S2, Vs_S2, Rho_S2, Rho_S1

def compute_relative_contrasts_interface_loop(Vp, Vs, rho):
    nz, nx = Vp.shape

    dVp_over_Vp = np.zeros((nz - 1, nx))
    dVs_over_Vs = np.zeros((nz - 1, nx))
    drho_over_rho = np.zeros((nz - 1, nx))

    for i in range(nx):
        for j in range(nz - 1):
            vp1, vp2 = Vp[j, i], Vp[j + 1, i]
            vs1, vs2 = Vs[j, i], Vs[j + 1, i]
            rho1, rho2 = rho[j, i], rho[j + 1, i]

            vp_avg = 0.5 * (vp1 + vp2)
            vs_avg = 0.5 * (vs1 + vs2)
            rho_avg = 0.5 * (rho1 + rho2)

            dVp_over_Vp[j, i] = (vp2 - vp1) / vp_avg
            dVs_over_Vs[j, i] = (vs2 - vs1) / vs_avg
            drho_over_rho[j, i] = (rho2 - rho1) / rho_avg

    # Add top row of zeros
    dVp_over_Vp = np.vstack([np.zeros((1, nx)), dVp_over_Vp])
    dVs_over_Vs = np.vstack([np.zeros((1, nx)), dVs_over_Vs])
    drho_over_rho = np.vstack([np.zeros((1, nx)), drho_over_rho])

    return dVp_over_Vp, dVs_over_Vs, drho_over_rho

def compute_RR(A, B, sin2, wav):
    nz, nx = A.shape
    ntheta = len(sin2)

    image_fit = np.zeros((ntheta, nz, nx), dtype=A.dtype)

    for i, s2 in enumerate(sin2):
        R_i = A + B * s2
        wavelet = wav[i]
        for j in range(nx):
            image_fit[i, :, j] = np.convolve(R_i[:, j], wavelet, mode='same')
    return image_fit

# Create output folder
os.makedirs("output_rr", exist_ok=True)

# Loop over 1-100
for idx in range(0, 801):
    fname = f"saturation_data/saturation_data_{idx:03d}.npy"
    Snw2 = np.load(fname)

    Vp_S2, Vs_S2, Rho_S2, _ = substitution(Por, Snw2, RhoS, RhoW, RhoNw, Vp_model, Vs_model)

    Vp_syn = np.vstack((Vp_over, Vp_S2[::-1]))
    Vs_syn = np.vstack((Vs_over, Vs_S2[::-1]))
    Rho_syn = np.vstack((Rho_over, Rho_S2[::-1]))

    dVp_over_Vp, dVs_over_Vs, drho_over_rho = compute_relative_contrasts_interface_loop(Vp_syn, Vs_syn, Rho_syn)

    Vs2_over_Vp2 = (Vs_syn**2) / (Vp_syn**2)
    A = 0.5 * (dVp_over_Vp + drho_over_rho)
    B = dVp_over_Vp - 2 * Vs2_over_Vp2 * (drho_over_rho + 2 * dVs_over_Vs)

    RR = compute_RR(A, B, sin2, wav)

    # Split and save each angle separately
    for angle_idx in range(RR.shape[0]):
        output_fname = f"output_rr/RR_{idx:03d}_angle{angle_idx}.npy"
        np.save(output_fname, RR[angle_idx])

    print(f"Processed and saved slices for saturation_data_{idx:03d}")
'''

import os, json, argparse, re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ---- import your physics functions ----
from physics import (
    compute_dRR_from_saturation_batch,
    compute_relative_contrasts,
    compute_RR,
)

# ---------------- helpers ----------------
def load_drr_tensor(data_dir, idx, angles=(0,1,2,3), baseline_idx=0, device=None):
    arrs = []
    for a in angles:
        y0 = np.load(os.path.join(data_dir, f"RR_{baseline_idx:03d}_angle{a}.npy"))
        y  = np.load(os.path.join(data_dir, f"RR_{idx:03d}_angle{a}.npy"))
        if y.shape != y0.shape:
            raise ValueError(f"angle {a} shape mismatch: {y.shape} vs {y0.shape}")
        arrs.append(y - y0)
    t = torch.tensor(np.stack(arrs, 0), dtype=torch.float32)
    return t.to(device) if device else t

def find_checkpoint(folder: str) -> str:
    mf = os.path.join(folder, "model_final.pth")
    if os.path.exists(mf): return mf
    pat = re.compile(r"epoch_(\d+)\.pth$")
    latest = None
    for f in os.listdir(folder):
        m = pat.match(f)
        if m:
            ep = int(m.group(1))
            if latest is None or ep > latest[0]:
                latest = (ep, f)
    if latest is None: raise FileNotFoundError("no ckpt")
    return os.path.join(folder, latest[1])

class DepthShift(torch.nn.Module):
    def __init__(self, tau_samples, max_shift=2.0, device=None):
        super().__init__()
        if tau_samples is None: tau_samples = torch.zeros(4, device=device)
        self.register_buffer("tau", tau_samples.clone().to(device))
        self.max_shift = float(max_shift)
    def forward(self, y):  # (B,4,H,W)
        if self.tau.abs().sum() == 0: return y
        B,C,H,W = y.shape
        ys = y.view(B*C,1,H,W)
        z = torch.linspace(-1,1,H, device=y.device, dtype=y.dtype).view(1,1,H,1).repeat(B*C,1,1,W)
        x = torch.linspace(-1,1,W, device=y.device, dtype=y.dtype).view(1,1,1,W).repeat(B*C,1,H,1)
        tau = torch.clamp(self.tau, -self.max_shift, self.max_shift)
        tz  = (2.0 * tau.repeat_interleave(B).view(B*C,1,1,1) / max(H-1,1))
        z = z + tz
        grid = torch.stack((x,z), dim=-1).squeeze(1)
        ys = F.grid_sample(ys, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return ys.view(B,C,H,W)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Forward-model sanity check from Snw")
    ap.add_argument("--idx", type=int, default=209)
    ap.add_argument("--data-dir", default="output_rr")
    ap.add_argument("--ckpt-dir", default="../output/2025-09-16_19-40")  # for depth shift
    ap.add_argument("--snw-path", default=None, help="Path to Snw_idxXXX.npy; if None, infer from ckpt-dir")
    ap.add_argument("--flip-z", type=str, default="auto", choices=["auto","true","false"])
    ap.add_argument("--zero-mean-wavelets", action="store_true", help="remove DC from wavelets before use")
    ap.add_argument("--use-calibration", action="store_true", help="read calibration.json for per-angle k")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- flip_z from training config if available ----
    if args.flip_z == "auto":
        try:
            with open(os.path.join(args.ckpt_dir, "config.json"), "r") as f:
                cfg = json.load(f)
            flip_z = bool(cfg.get("flip_z", True))
            print("[cfg] flip_z =", flip_z)
        except Exception:
            flip_z = True
            print("[cfg] config.json missing; default flip_z=True")
    else:
        flip_z = (args.flip_z == "true")

    # ---- load Snw ----
    if args.snw_path is None:
        args.snw_path = os.path.join(args.ckpt_dir, "inference", f"Snw_idx{args.idx:03d}.npy")
    Snw_np = np.load(args.snw_path).astype(np.float32)  # (H,W)
    Snw = torch.tensor(Snw_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    Snw = Snw.clamp(0.01, 0.99)
    print("Snw:", Snw.shape, f"min={Snw.min().item():.3f} max={Snw.max().item():.3f}")

    # ---- load backgrounds ----
    def maybe_flip(t):
        return torch.flip(t, dims=[0]).contiguous() if flip_z else t
    Vp  = torch.tensor(np.load("Vp_syn.npy").astype(np.float32))
    Vs  = torch.tensor(np.load("Vs_syn.npy").astype(np.float32))
    Rho = torch.tensor(np.load("Rho_syn.npy").astype(np.float32))
    Por = torch.tensor(np.load("Porosity.npy").astype(np.float32))
    Vp,Vs,Rho,Por = map(maybe_flip, (Vp,Vs,Rho,Por))
    Vp,Vs,Rho,Por = Vp.to(device), Vs.to(device), Rho.to(device), Por.to(device)

    # ---- wavelets & sin^2 ----
    wavelets = []
    for i in range(4):
        w = torch.tensor(np.load(f"wavelet_{i}.npy").astype(np.float32), device=device)
        if args.zero_mean_wavelets:
            w = w - w.mean()
        wavelets.append(w)
    sin2 = [0.0, 0.1, 0.2, 0.3]
    if args.use_calibration and os.path.exists("calibration.json"):
        try:
            with open("calibration.json","r") as f:
                k = json.load(f).get("k", None)
            if k and len(k)==4:
                sin2 = k
                print("[calibration] using k =", k)
        except Exception:
            pass

    # ---- build baseline RR0 from backgrounds/wavelets ----
    with torch.no_grad():
        dVp0, dVs0, dRho0 = compute_relative_contrasts(Vp, Vs, Rho)
        ratio0 = (Vs**2) / (Vp**2 + 1e-8)
        A0 = 0.5 * (dVp0 + dRho0)
        B0 = dVp0 - 2.0 * ratio0 * (dRho0 + 2.0 * dVs0)
        RR0 = compute_RR(A0, B0, sin2, wavelets)  # (4,H,W)

    # ---- observed dRR from files ----
    dRR_obs = load_drr_tensor(args.data_dir, args.idx, device=device).unsqueeze(0)  # (1,4,H,W)

    # ---- optional depth shift from ckpt ----
    depth_shift_tau = None
    try:
        ckpt = torch.load(find_checkpoint(args.ckpt_dir), map_location=device)
        dst = ckpt.get("depth_shift_tau", None)
        if dst is not None:
            depth_shift_tau = dst.to(device)
            print("Loaded depth shifts:", depth_shift_tau.detach().cpu().numpy())
    except Exception:
        pass
    shift_layer = DepthShift(depth_shift_tau, max_shift=2.0, device=device).eval()

    # ---- physics forward from Snw ----
    with torch.no_grad():
        M = (Por > 0.05).float().unsqueeze(0).unsqueeze(0)       # same mask as train
        Snw_phys = F.avg_pool2d((Snw * M).contiguous(), 3, 1, 1)
        RR_pred = compute_dRR_from_saturation_batch(Snw_phys, Vp, Vs, Por, sin2, wavelets=wavelets)  # (1,4,H,W)
        RR_pred = shift_layer(RR_pred)
        P = RR_pred - RR0.unsqueeze(0)                           # (1,4,H,W)

        # analytic sign + gain vs observed dRR
        numer_corr = (P * dRR_obs).mean(dim=(0,2,3))
        sign = torch.where(numer_corr >= 0, 1.0, -1.0).to(P.dtype)
        P_eff = sign.view(1,4,1,1) * P
        numer = (P_eff * dRR_obs).mean(dim=(0,2,3))
        denom = (P_eff * P_eff).mean(dim=(0,2,3)).clamp_min(1e-8)
        amp = (numer/denom).detach()
        dRR_hat = amp.view(1,4,1,1) * P_eff

        resid = (dRR_hat - dRR_obs).squeeze(0).detach().cpu().numpy()  # (4,H,W)
        rmse_per_angle = np.sqrt((resid**2).mean(axis=(1,2)))
        overall = float(np.sqrt((resid**2).mean()))
        print("Analytic amp:", amp.detach().cpu().numpy(), " sign:", sign.cpu().numpy())
        print("Physics-check RMSE per angle:", rmse_per_angle, " overall:", overall)

        # baseline mismatch diagnostic
        RR_base = []
        for i in range(4):
            RR_base.append(np.load(os.path.join(args.data_dir, f"RR_000_angle{i}.npy")).astype(np.float32))
        RR_base = torch.tensor(np.stack(RR_base,0), device=device)
        base_mis = (RR_base - RR0).detach().cpu().numpy()
        base_rmse = np.sqrt((base_mis**2).mean(axis=(1,2)))
        print("[baseline check] RMSE per angle between dataset RR_base and computed RR0:", base_rmse)

        # save residual figure
        out_dir = os.path.join(args.ckpt_dir, "inference")
        os.makedirs(out_dir, exist_ok=True)
        fig, axs = plt.subplots(2,2, figsize=(11,5.5), constrained_layout=True)
        for i, ax in enumerate(axs.ravel()):
            vmax = np.percentile(np.abs(resid[i]), 99.5)
            im = ax.imshow(resid[i], origin="upper", cmap="seismic",
                           vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"Residual angle {i} (RMSE={rmse_per_angle[i]:.3e})")
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"Residual dRR (idx {args.idx:03d})")
        fig.savefig(os.path.join(out_dir, f"residual_dRR_idx{args.idx:03d}.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    main()
