# unet.py
import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ------------------------------
# Basic building blocks
# ------------------------------

class ResConv(nn.Module):
    """Two 3x3 convs with residual (projection if channels change)."""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.block(x)
        s = self.proj(x)
        return self.act(y + s)


class DoubleConv(nn.Module):
    """Your original non-residual block."""
    def __init__(self, in_channels, out_channels, filter_size=3):
        super().__init__()
        p = filter_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter_size, 1, p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, filter_size, 1, p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ------------------------------
# Simple positional encoding for 2D
# ------------------------------

def sine_posenc_2d(h, w, dim, device):
    """(h*w, dim) 2D sine-cos positional encoding."""
    assert dim % 4 == 0, "posenc dim must be multiple of 4"
    y = torch.linspace(0, 1, steps=h, device=device)
    x = torch.linspace(0, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # (h,w)
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)
    d = dim // 4
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), d, device=device))
    y_sin = torch.sin(yy * freqs)
    y_cos = torch.cos(yy * freqs)
    x_sin = torch.sin(xx * freqs)
    x_cos = torch.cos(xx * freqs)
    pe = torch.cat([y_sin, y_cos, x_sin, x_cos], dim=1)  # (h*w, dim)
    return pe

class TransformerBottleneck(nn.Module):
    """Tiny ViT-style encoder used at the U-Net bottleneck."""
    def __init__(self, dim, num_heads=4, num_layers=2, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,          # (B, N, C)
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B,C,H,W) -> (B,HW,C)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        pe = sine_posenc_2d(H, W, C, x.device)
        x_flat = x_flat + pe.unsqueeze(0)
        y = self.enc(x_flat)               # (B, HW, C)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y

# ------------------------------
# UNet variants
# ------------------------------

class UNET(nn.Module):
    """Original UNet (kept for reference)."""
    def __init__(self, in_channels=1, out_channels=1, filter_size=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f, filter_size))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2, filter_size)

        prev = features[-1]*2
        for f in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev, f, kernel_size=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            self.ups.append(DoubleConv(f*2, f, filter_size))
            prev = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, -2.0)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i//2]
            if x.shape[2:] != s.shape[2:]:
                x = TF.resize(x, s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i+1](x)
        return self.final_conv(x)


class UNET_Res(nn.Module):
    """UNet with residual conv blocks (encoder/decoder)."""
    def __init__(self, in_channels=1, out_channels=1, filter_size=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(ResConv(ch, f, k=filter_size))
            ch = f

        self.bottleneck = ResConv(features[-1], features[-1]*2, k=filter_size)

        prev = features[-1]*2
        for f in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev, f, kernel_size=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            self.ups.append(ResConv(f*2, f, k=filter_size))
            prev = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, -2.0)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i//2]
            if x.shape[2:] != s.shape[2:]:
                x = TF.resize(x, s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i+1](x)
        return self.final_conv(x)


class UNET_TF_Res(nn.Module):
    """UNet with residual conv blocks + small transformer at bottleneck."""
    def __init__(
        self,
        in_channels=1, out_channels=1, filter_size=3, features=[64, 128, 256, 512],
        tf_heads=4, tf_layers=2, tf_mlp_ratio=4.0, tf_dropout=0.0,
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(ResConv(ch, f, k=filter_size))
            ch = f

        bottleneck_dim = features[-1]*2
        self.bottleneck = ResConv(features[-1], bottleneck_dim, k=filter_size)
        self.tf = TransformerBottleneck(dim=bottleneck_dim, num_heads=tf_heads,
                                        num_layers=tf_layers, mlp_ratio=tf_mlp_ratio,
                                        dropout=tf_dropout)

        prev = bottleneck_dim
        for f in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(prev, f, kernel_size=1, bias=False),
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
            ))
            self.ups.append(ResConv(f*2, f, k=filter_size))
            prev = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        if self.final_conv.bias is not None:
            nn.init.constant_(self.final_conv.bias, -2.0)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.tf(x)                   # transformer at 1/16 resolution (with features=[64,128,256,512])

        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i//2]
            if x.shape[2:] != s.shape[2:]:
                x = TF.resize(x, s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i+1](x)

        return self.final_conv(x)
