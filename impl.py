# impl.py
import os, glob, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from physics import compute_dRR_from_saturation_batch


# --------------------------- small utils ---------------------------

def zscore(X, mu, sigma):
    """Channel-wise standardization using μ/σ (shape (4,))."""
    mu = mu.view(1, -1, 1, 1)
    sigma = sigma.view(1, -1, 1, 1)
    return (X - mu) / (sigma + 1e-8)


def sobel_xy(x):
    """Depth(x)/Lateral(z) Sobel grads per-channel."""
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[ 1,  2,  1],
                       [ 0,  0,  0],
                       [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    kx = kx.repeat(x.shape[1], 1, 1, 1)
    ky = ky.repeat(x.shape[1], 1, 1, 1)
    px = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    py = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return px, py


def tv_aniso(x, wx=1.0, wz=1.5):
    """Anisotropic TV (slightly stronger along depth by default)."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dz = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (wx * dx.abs().mean() + wz * dz.abs().mean())


def grad_l2(x):
    """L2 of Sobel gradients."""
    gx, gz = sobel_xy(x)
    return (gx.pow(2).mean() + gz.pow(2).mean())


def manage_checkpoints(folder, max_keep=5):
    """Keep only the newest max_keep epoch_*.pth files."""
    paths = []
    for p in glob.glob(os.path.join(folder, "epoch_*.pth")):
        try:
            ep = int(os.path.basename(p).split("_")[1].split(".")[0])
            paths.append((ep, p))
        except Exception:
            pass
    paths.sort()
    while len(paths) > max_keep:
        _, old = paths.pop(0)
        try:
            os.remove(old)
            print(f"Removed old checkpoint: {os.path.basename(old)}")
        except Exception:
            pass


# --------------------------- masked losses ---------------------------

def masked_ms_loss(A, B, WM):
    """
    Scale-invariant masked MSE over all pixels & channels.
    A,B: (B,C,H,W); WM: (B,C,H,W) or (1,1,H,W)
    """
    if WM.dim() == 4 and WM.shape[1] == 1:
        WM = WM.expand_as(A)
    res = (A - B)
    num = (res.pow(2) * WM).sum()
    den = (WM.sum() * A.shape[1]).clamp_min(1.0)  # mean over mask × channels
    return num / den


def masked_grad_loss(A, B, WM):
    """Masked L1 discrepancy of Sobel grads, normalized like masked_ms_loss."""
    if WM.dim() == 4 and WM.shape[1] == 1:
        WM = WM.expand_as(A)
    Ax, Az = sobel_xy(A)
    Bx, Bz = sobel_xy(B)
    num = ((Ax - Bx).abs() * WM).sum() + ((Az - Bz).abs() * WM).sum()
    den = (WM.sum() * A.shape[1]).clamp_min(1.0)
    return num / den


# ---------- stripe (row-constant) projection helper ----------

def project_out_row_constant(P, WM, target=None, orth_on_target=True):
    """
    Remove the per-depth (row-constant across x) component from P.
    If target is given and orth_on_target=True, remove the *same component*
    from target so the supervision does not penalize it.

    P: (B,4,H,W); WM: (B,4,H,W) or (1,1,H,W)
    Returns: P_orth, target_orth, coef  where coef is (B,4,1,1)
    """
    if WM.dim() == 4 and WM.shape[1] == 1:
        WM = WM.expand_as(P)

    # Build row-constant template from masked row mean
    wsum = WM.sum(dim=3, keepdim=True).clamp_min(1e-6)           # (B,4,H,1)
    row_mean = (P * WM).sum(dim=3, keepdim=True) / wsum           # (B,4,H,1)
    Btemp = row_mean.expand_as(P)                                  # (B,4,H,W)

    # Least-squares coefficient under the mask
    denom = (Btemp.pow(2) * WM).sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)  # (B,4,1,1)
    numer = (P * Btemp * WM).sum(dim=(2, 3), keepdim=True)                      # (B,4,1,1)
    coef = numer / denom                                                        # (B,4,1,1)

    P_orth = P - coef * Btemp
    if (target is not None) and orth_on_target:
        target_orth = target - coef * Btemp
    else:
        target_orth = target
    return P_orth, target_orth, coef


# --------------------- fractional depth shift ---------------------

class DepthShift(nn.Module):
    """Small, learnable vertical shift (per angle) using grid_sample."""
    def __init__(self, n_angles=4, max_shift=2.0):
        super().__init__()
        self.tau = nn.Parameter(torch.zeros(n_angles))
        self.max_shift = float(max_shift)

    def forward(self, y):  # y: (B,4,H,W)
        if self.tau.detach().abs().sum() < 1e-12:
            return y
        B, C, H, W = y.shape
        ys = y.view(B * C, 1, H, W)
        z = torch.linspace(-1, 1, H, device=y.device, dtype=y.dtype).view(1, 1, H, 1).repeat(B * C, 1, 1, W)
        x = torch.linspace(-1, 1, W, device=y.device, dtype=y.dtype).view(1, 1, 1, W).repeat(B * C, 1, H, 1)
        tz = (2.0 * torch.clamp(self.tau, -self.max_shift, self.max_shift)
              .repeat_interleave(B).view(B * C, 1, 1, 1) / max(H - 1, 1))
        grid = torch.stack((x, z + tz), dim=-1).squeeze(1)
        ys = F.grid_sample(ys, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return ys.view(B, C, H, W)


# ------------------- depth-wise bias detrending -------------------

def _smooth_depth_only(t, k=81):
    if k <= 1:
        return t
    pad = (k // 2, 0)
    return F.avg_pool2d(t, kernel_size=(k, 1), stride=1, padding=pad)


def detrend_depthwise(Y, M, k=81):
    """
    Remove per-depth (column) mean inside mask M from Y.
    Y: (B,4,H,W), M: (1,1,H,W)
    Returns Y_detr and the bias map b (B,4,H,1)
    """
    B, C, H, W = Y.shape
    WM = M.expand(B, C, H, W)
    wsum = WM.sum(dim=3, keepdim=True).clamp_min(1e-6)  # (B,4,H,1)
    b = (Y * WM).sum(dim=3, keepdim=True) / wsum        # (B,4,H,1)
    b = _smooth_depth_only(b, k=k)
    Y_detr = Y - b.repeat(1, 1, 1, W)
    return Y_detr, b


# -------------------------- training loop --------------------------

def train_dnn(
    model, train_loader, device, criterion_type, optimizer_type,
    learning_rate, weight_decay, max_epochs, dirs,
    Vp_syn, Vs_syn, Por_syn, RR0_cached, wavelets, sin2,
    *,
    mu, sigma,
    M,
    max_keep=5,
    lam_tv=1e-2, lam_edge=5e-3, lam_out=5e-1, lam_sps=5e-4,
    shift_reg=1e-3, wavelet_reg=1e-3,     # wavelet_reg kept for API parity (unused)
    angle_weights=None, use_amp=True, grad_clip=1.0,
    log_every=50, smooth_snw=True,
    detrend_k=81,
    # ▼ NEW stripe knobs
    orth_project=False,
    orth_on_target=True,
    baseline_coef_weight=1e-2,
):
    """
    Physics-guided training (flip-free), with:
      - depth-window mask + taper
      - depth-wise detrend of input and target
      - per-angle sign + analytic amplitude
      - optional projection away from row-constant (stripe) component
    """
    model.to(device)
    RR0_cached = RR0_cached.unsqueeze(0).contiguous().to(device)  # (1,4,H,W)
    mu, sigma = mu.to(device), sigma.to(device)
    M = M.to(device)
    wavelets = [w.to(device) for w in wavelets]

    # Optimizer
    if optimizer_type.upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.upper() == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer type")

    # Trainable fractional depth shift (regularized)
    shift_layer = DepthShift(n_angles=4, max_shift=2.0).to(device)
    optimizer.add_param_group({"params": [shift_layer.tau], "lr": learning_rate * 5})

    # Scheduler / AMP
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        def autocast_ctx(): return torch.amp.autocast('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        def autocast_ctx(): return torch.cuda.amp.autocast(enabled=use_amp)

    train_losses, global_step = [], 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            global_step += 1
            if batch is None:
                continue

            RR_input = batch[0] if isinstance(batch, (tuple, list)) else batch   # (B,4,H,W)
            RR_input = RR_input.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                # --- detrend input/target identically ---
                RR_detr, _ = detrend_depthwise(RR_input, M, k=detrend_k)  # (B,4,H,W)
                RR_target_detr = RR_detr

                # forward net -> Snw in [0,1]
                logits = model(RR_detr)
                Snw = torch.sigmoid(logits).clamp(0.01, 0.99)

                # mask + optional light smoothing (for physics forward only)
                Snw_phys = (Snw * M).contiguous()
                if smooth_snw:
                    Snw_phys = F.avg_pool2d(Snw_phys, kernel_size=3, stride=1, padding=1)

                # physics forward
                RR_pred = compute_dRR_from_saturation_batch(
                    Snw_phys, Vp_syn, Vs_syn, Por_syn, sin2, wavelets=wavelets
                )  # (B,4,H,W)
                RR_pred = shift_layer(RR_pred)
                loss_shift = shift_reg * (shift_layer.tau ** 2).mean()

                # residual w.r.t. baseline files and mask
                P = RR_pred - RR0_cached                                # (B,4,H,W)
                WM = M.expand_as(P)

                # --- optional stripe projection (row-constant per row) ---
                if orth_project:
                    P_base, T_base, coef = project_out_row_constant(
                        P, WM, target=RR_target_detr, orth_on_target=orth_on_target
                    )
                    # tiny Tikhonov on removed component
                    L_bg = baseline_coef_weight * coef.pow(2).mean()
                else:
                    P_base, T_base = P, RR_target_detr
                    L_bg = Snw.new_tensor(0.0)

                # per-angle sign using projected quantities
                numer_corr = ((P_base * T_base) * WM).sum(dim=(0, 2, 3))
                sign = torch.where(numer_corr >= 0, 1.0, -1.0).to(P.dtype)
                P_eff = sign.view(1, 4, 1, 1) * P_base

                # optional angle weights
                if angle_weights is not None:
                    w = angle_weights.view(1, 4, 1, 1)
                    Pw = P_eff * w
                    Yw = T_base * w
                else:
                    Pw = P_eff
                    Yw = T_base

                # analytic amplitude on the (projected) residual
                numer = (Pw * Yw * WM).sum(dim=(0, 2, 3))
                denom = (Pw.pow(2) * WM).sum(dim=(0, 2, 3)).clamp_min(1e-8)
                amp_vec = (numer / denom).detach()                      # (4,)

                dRR_pred = amp_vec.view(1, 4, 1, 1) * Pw                # (B,4,H,W)

                # standardize with μ/σ from detrended data
                Xp = zscore(dRR_pred, mu, sigma)
                X  = zscore(T_base,     mu, sigma)

                # masked, scale-invariant losses
                L_data = masked_ms_loss(Xp, X, WM)
                L_grad = masked_grad_loss(Xp, X, WM)

                # regularizers on Snw
                L_tv   = tv_aniso(Snw)
                L_edge = grad_l2(Snw)
                L_out  = (Snw * (1.0 - M)).mean()
                L_sps  = Snw.mean()

                loss = (L_data + 0.1 * L_grad +
                        lam_tv * L_tv + lam_edge * L_edge +
                        lam_out * L_out + lam_sps * L_sps +
                        loss_shift + L_bg)

            # occasional log (corr before sign to show innate polarity)
            if (global_step % log_every) == 0:
                denom_corr = (
                    torch.sqrt((P_eff.pow(2)  * WM).sum(dim=(0, 2, 3)).clamp_min(1e-8)) *
                    torch.sqrt((T_base.pow(2) * WM).sum(dim=(0, 2, 3)).clamp_min(1e-8))
                ).clamp_min(1e-8)
                corr_val = numer_corr / denom_corr
                logging.info(f"corr per angle: {np.round(corr_val.detach().cpu().numpy(), 3)}")
                logging.info(
                    "epoch=%d step=%d  L=%.3e | data=%.3e grad=%.3e bg=%.3e | tv=%.3e edge=%.3e out=%.3e sps=%.3e | "
                    "sign=%s amp=%s",
                    epoch + 1, global_step, float(loss.detach()),
                    float(L_data.detach()), float(L_grad.detach()), float(L_bg.detach()),
                    float(L_tv.detach()), float(L_edge.detach()), float(L_out.detach()), float(L_sps.detach()),
                    str([float(s) for s in sign]),
                    str([float(a) for a in amp_vec])
                )

            # backward
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                params = [p for g in optimizer.param_groups for p in g["params"]
                          if (p is not None and p.requires_grad)]
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach())

        epoch_loss /= max(1, len(train_loader))
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {epoch_loss:.6f}")
        scheduler.step(epoch_loss)

        lr = optimizer.param_groups[0]["lr"]
        try:
            print(f"epoch={epoch+1} loss={epoch_loss:.6f} lr={lr:.2e}  shift_tau={shift_layer.tau.detach().cpu().numpy()}")
        except Exception:
            pass

        torch.save({
            "model_state_dict": model.state_dict(),
            "depth_shift_tau": shift_layer.tau.detach().cpu(),
        }, os.path.join(dirs, f"epoch_{epoch+1}.pth"))
        manage_checkpoints(dirs, max_keep=max_keep)

    return train_losses, model
