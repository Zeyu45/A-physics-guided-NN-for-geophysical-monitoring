# run.py
import os, json, argparse, logging
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataio import RRDataset, list_rr_indices
from unet import UNET, UNET_Res, UNET_TF_Res
from impl import train_dnn, detrend_depthwise   # must match training detrend

# ----------------- CLI -----------------
def get_args():
    p = argparse.ArgumentParser(description="Train inverse NN with physics-guided loss (RR0 from files)")
    p.add_argument("--data-dir", default="output_rr")

    p.add_argument("--net", default="unet_tf_res",
                   choices=["unet", "unet_tf_res", "unet_res"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)

    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="../output")

    p.add_argument("--zero-mean-wavelets", dest="zero_mean_wavelets", action="store_true")
    p.add_argument("--no-zero-mean-wavelets", dest="zero_mean_wavelets", action="store_false")
    p.set_defaults(zero_mean_wavelets=True)

    p.add_argument("--keep-checkpoints", type=int, default=5)

    # mask + detrend hyperparams
    p.add_argument("--zmin", type=int, default=160)
    p.add_argument("--zmax", type=int, default=340)
    p.add_argument("--taper", type=int, default=24)   # cosine taper rows at top/bottom
    p.add_argument("--detrend-k", type=int, default=81)

    # stripe handling knobs (match impl.py)
    p.add_argument("--orth-project", dest="orth_project", action="store_true")
    p.add_argument("--no-orth-project", dest="orth_project", action="store_false")
    p.set_defaults(orth_project=True)
    p.add_argument("--orth-on-target", dest="orth_on_target", action="store_true")
    p.add_argument("--no-orth-on-target", dest="orth_on_target", action="store_false")
    p.set_defaults(orth_on_target=True)
    p.add_argument("--baseline-coef-weight", type=float, default=1e-2)

    return p.parse_args()

# ----------------- logging -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------- helpers -----------------
def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)

def timestamp_dir(root):
    t = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out = os.path.join(root, t); os.makedirs(out, exist_ok=True); return out

def collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)

def depth_window_mask(H, W, zmin, zmax, taper, device, dtype):
    """
    Cosine-tapered rectangular mask along depth (z), identical to inference.
    """
    z = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    core = ((z >= zmin) & (z <= zmax)).float()
    if taper > 0:
        top = torch.clamp(z - zmin, 0, taper) / max(taper, 1)
        bot = torch.clamp(zmax - z, 0, taper) / max(taper, 1)
        edge = torch.minimum(top, bot)                 # 0..1 within taper zones
        taper_w = 0.5 * (1 - torch.cos(np.pi * edge)) # smooth 0→1→0
        core = core * taper_w
    return core.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

@torch.no_grad()
def compute_channel_stats_detrended(loader, device, M, detrend_k=81):
    """
    Masked μ/σ computed on the *detrended* data (same detrend as training).
    """
    M = M.to(device)
    s = torch.zeros(4, device=device)
    ss = torch.zeros(4, device=device)
    wsum = torch.zeros(4, device=device)

    for batch in loader:
        if batch is None:
            continue
        rr = batch[0] if isinstance(batch, (tuple, list)) else batch  # (B,4,H,W)
        rr = rr.to(device, non_blocking=True)

        rr_detr, _ = detrend_depthwise(rr, M, k=detrend_k)            # (B,4,H,W)
        WM = M.expand_as(rr_detr)

        s  += (rr_detr * WM).sum(dim=(0, 2, 3))
        ss += (rr_detr * rr_detr * WM).sum(dim=(0, 2, 3))
        wsum += WM.sum(dim=(0, 2, 3))

    mu = s / wsum.clamp_min(1.0)
    var = (ss / wsum.clamp_min(1.0)) - mu * mu
    sigma = torch.sqrt(var.clamp_min(1e-12))
    return mu, sigma

def list_rr_indices_sorted(data_dir):
    return sorted(list_rr_indices(data_dir))

def discover_indices(data_dir, angles, baseline_shape):
    all_idx = list_rr_indices_sorted(data_dir)
    def ok(i):
        for a in angles:
            f = os.path.join(data_dir, f"RR_{i:03d}_angle{a}.npy")
            if not os.path.exists(f): return False
            if np.load(f, mmap_mode="r").shape != baseline_shape: return False
        return i not in [0, 800]
    valid = [i for i in all_idx if ok(i)]
    return valid if valid else [i for i in all_idx if i != 0]

# ----------------- main -----------------
def main():
    args = get_args(); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    out_dir = timestamp_dir(args.out_root); print("Output folder:", out_dir)

    # physics constants
    sin2 = [0.0, 0.1, 0.2, 0.3]; angles = [0, 1, 2, 3]

    # ------------- baseline RR0_files -------------
    base_paths = [os.path.join(args.data_dir, f"RR_000_angle{a}.npy") for a in angles]
    for p in base_paths:
        if not os.path.exists(p): raise FileNotFoundError(f"Missing baseline: {p}")
    base = [np.load(p, mmap_mode="r") for p in base_paths]
    if any(b.shape != base[0].shape for b in base):
        raise ValueError(f"Baseline angle shapes differ: {[b.shape for b in base]}")
    baseline_shape = base[0].shape
    RR0_cached = torch.tensor(np.stack(base, axis=0), dtype=torch.float32, device=device)  # (4,H,W)

    train_indices = discover_indices(args.data_dir, angles, baseline_shape)
    print(f"[data] using {len(train_indices)} snapshots")

    # ------------- dataset / loaders (RR_i - RR_000) -------------
    train_dataset = RRDataset(args.data_dir, train_indices, por_path=None)

    # mask (exactly as used in training & inference)
    H, W = baseline_shape
    M = depth_window_mask(H, W, args.zmin, args.zmax, args.taper, device=device, dtype=torch.float32)
    print(f"[mask] depth window z∈[{args.zmin},{args.zmax}] with taper={args.taper}")

    # stats loader for μ/σ
    stats_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"), collate_fn=collate_drop_none,
    )

    # μ/σ from DETRENDED + MASKED data (must match training)
    mu, sigma = compute_channel_stats_detrended(stats_loader, device, M, detrend_k=1)
    logging.info(f"mu={mu.tolist()}")
    logging.info(f"sigma={sigma.tolist()}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        pin_memory=(device.type == "cuda"), collate_fn=collate_drop_none,
    )

    # quick channel correlation on one sample (sanity)
    with torch.no_grad():
        sample = train_dataset[0]
        sample = sample[0] if isinstance(sample, (tuple, list)) else sample
        rr = sample.unsqueeze(0).to(device)  # (1,4,H,W)
        C = rr.shape[1]
        X = rr.view(C, -1) - rr.view(C, -1).mean(dim=1, keepdim=True)
        cov = (X @ X.t()) / (X.shape[1] - 1 + 1e-8)
        d = torch.sqrt(torch.diag(cov) + 1e-12)
        corr = cov / (d.view(-1, 1) * d.view(1, -1) + 1e-12)
        print("Channel–channel correlation (single snapshot):\n", corr)

    # ------------- backgrounds & wavelets -------------
    Por = torch.tensor(np.load("Porosity.npy").astype(np.float32), device=device)
    Vp  = torch.tensor(np.load("Vp_syn.npy").astype(np.float32),   device=device)
    Vs  = torch.tensor(np.load("Vs_syn.npy").astype(np.float32),   device=device)

    wavelets = []
    for i in range(4):
        w = np.load(f"wavelet_{i}.npy").astype(np.float32)
        if args.zero_mean_wavelets: w = w - w.mean()
        wavelets.append(torch.tensor(w, device=device))

    # ------------- model -------------
    in_channels, out_channels = 4, 1
    if args.net == "unet":
        model = UNET(in_channels=in_channels, out_channels=out_channels, filter_size=3, features=[64,128,256,512])
        net_name = "UNet"
    elif args.net == "unet_tf_res":
        try:
            model = UNET_TF_Res(in_channels=in_channels, out_channels=out_channels,
                                filter_size=3, features=[64,128,256,512],
                                tf_layers=2, tf_heads=8, tf_mlp_ratio=4.0, tf_dropout=0.0)
        except TypeError:
            model = UNET_TF_Res(in_channels=in_channels, out_channels=out_channels,
                                filter_size=3, features=[64,128,256,512],
                                tf_depth=2, tf_heads=8, tf_mlp_ratio=4.0, tf_dropout=0.0)
        net_name = "UNet (res blocks + Transformer)"
    elif args.net == "unet_res":
        model = UNET_Res(in_channels=in_channels, out_channels=out_channels, filter_size=3, features=[64,128,256,512])
        net_name = "UNet (res blocks)"
    else:
        raise ValueError(f"Unknown net: {args.net}")

    print(f"Device: {device}")
    print(f"Network: {net_name}")
    print(f"LR={args.lr}, WD={args.wd}, batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"[train] zero_mean_wavelets={args.zero_mean_wavelets} | RR0_source=files | flip-free")

    # ------------- persist config -------------
    cfg = {
        "data_dir": args.data_dir,
        "net": args.net,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wd": args.wd,
        "amp": bool(args.amp),
        "seed": args.seed,
        "out_root": args.out_root,
        "flip_z": False,
        "zero_mean_wavelets": bool(args.zero_mean_wavelets),
        "rr0_source": "files",
        "keep_checkpoints": args.keep_checkpoints,
        "sin2": [0.0, 0.1, 0.2, 0.3],
        "mask": {"zmin": args.zmin, "zmax": args.zmax, "taper": args.taper},
        "detrend_k": args.detrend_k,
        "orth_project": bool(args.orth_project),
        "orth_on_target": bool(args.orth_on_target),
        "baseline_coef_weight": float(args.baseline_coef_weight),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ------------- train -------------
    use_amp = bool(args.amp and (device.type == "cuda"))
    train_losses, _ = train_dnn(
        model, train_loader, device,
        criterion_type="MSE", optimizer_type="Adam",
        learning_rate=args.lr, weight_decay=args.wd,
        max_epochs=args.epochs, dirs=out_dir,

        # physics guidance inputs
        Vp_syn=Vp, Vs_syn=Vs, Por_syn=Por,
        RR0_cached=RR0_cached, wavelets=wavelets, sin2=sin2,

        # dataset stats & mask (DETREND μ/σ!)
        mu=mu, sigma=sigma, M=M,

        # trainer knobs
        max_keep=args.keep_checkpoints, use_amp=use_amp, grad_clip=1.0,
        log_every=50, smooth_snw=True,

        # regularization
        lam_tv=1e-3, lam_edge=0.0, lam_out=0.3, lam_sps=0.0,
        shift_reg=0.0, wavelet_reg=0.0,

        detrend_k=1,

        # stripe handling
        orth_project=False,
        orth_on_target=False,
        baseline_coef_weight=False,
    )

    # ------------- save loss curve & final ckpt -------------
    np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses, dtype=np.float32))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.title("Training Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    loss_plot_path = os.path.join(out_dir, "loss_plot.png")
    plt.savefig(loss_plot_path, dpi=150); plt.close()
    print(f"Loss curve saved: {loss_plot_path}")

    # copy final
    src = os.path.join(out_dir, f"epoch_{args.epochs}.pth")
    dst = os.path.join(out_dir, "model_final.pth")
    try:
        import shutil; shutil.copyfile(src, dst)
        print(f"Saved final checkpoint from epoch_{args.epochs}.pth -> model_final.pth")
    except Exception as e:
        print(f"[warn] could not copy final ckpt: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
