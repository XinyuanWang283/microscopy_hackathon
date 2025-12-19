"""
Train a simple conditional SR3 diffusion model on MoS2 patches.

SR3 (LR conditioned super-resolution) is kept
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image


# --------------------------
# Utilities
# --------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, device=device) / float(half_dim - 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def make_gaussian_kernel2d(sigma: float, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        return None
    radius = int(3 * sigma)
    size = 2 * radius + 1
    coords = torch.arange(size, device=device) - radius
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[None, None]


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor


def make_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


# --------------------------
# Model
# --------------------------


class TimeBlock(nn.Module):
    """Legacy conv block matching existing checkpoints (time_dim=64, base=32)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_emb = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h


class LegacyUNet(nn.Module):
    """
    Architecture aligned with existing checkpoints in sr3_runs/mos2_sr3_fast_rerun:
    - time embedding dim 64
    - base channels 32
    - out conv 1x1
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.ModuleDict(
            dict(
                conv1=nn.Conv2d(in_channels, base_channels, 3, padding=1),
                norm1=nn.GroupNorm(8, base_channels),
                conv2=nn.Conv2d(base_channels, base_channels, 3, padding=1),
                norm2=nn.GroupNorm(8, base_channels),
                time_emb=nn.Linear(time_dim, base_channels),
                act=nn.SiLU(),
            )
        )

        # Down: use stride=2 in conv2 to downsample
        self.down1 = nn.ModuleDict(
            dict(
                conv1=nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
                norm1=nn.GroupNorm(8, base_channels * 2),
                conv2=nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, stride=2),
                norm2=nn.GroupNorm(8, base_channels * 2),
                time_emb=nn.Linear(time_dim, base_channels * 2),
                act=nn.SiLU(),
            )
        )
        self.down2 = nn.ModuleDict(
            dict(
                conv1=nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
                norm1=nn.GroupNorm(8, base_channels * 4),
                conv2=nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, stride=2),
                norm2=nn.GroupNorm(8, base_channels * 4),
                time_emb=nn.Linear(time_dim, base_channels * 4),
                act=nn.SiLU(),
            )
        )

        self.mid = nn.ModuleDict(
            dict(
                conv1=nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                norm1=nn.GroupNorm(8, base_channels * 4),
                conv2=nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                norm2=nn.GroupNorm(8, base_channels * 4),
                time_emb=nn.Linear(time_dim, base_channels * 4),
                act=nn.SiLU(),
            )
        )

        self.up2 = nn.ModuleDict(
            dict(
                trans=nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, stride=2, padding=1),
                block=TimeBlock(base_channels * 2, base_channels * 2, time_dim, groups=8),
            )
        )
        self.up1 = nn.ModuleDict(
            dict(
                trans=nn.ConvTranspose2d(base_channels * 4, base_channels, 4, stride=2, padding=1),
                block=TimeBlock(base_channels, base_channels, time_dim, groups=8),
            )
        )
        self.out = nn.Conv2d(base_channels, 1, 1)

    def stem_forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s = self.stem
        h = s["conv1"](x)
        h = s["norm1"](h)
        h = s["act"](h)
        h = h + s["time_emb"](t_emb)[:, :, None, None]
        h = s["conv2"](h)
        h = s["norm2"](h)
        h = s["act"](h)
        return h

    def down_forward(self, block: nn.ModuleDict, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = block["conv1"](x)
        h = block["norm1"](h)
        h = block["act"](h)
        h = h + block["time_emb"](t_emb)[:, :, None, None]
        h = block["conv2"](h)
        h = block["norm2"](h)
        h = block["act"](h)
        return h

    def mid_forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        m = self.mid
        h = m["conv1"](x)
        h = m["norm1"](h)
        h = m["act"](h)
        h = h + m["time_emb"](t_emb)[:, :, None, None]
        h = m["conv2"](h)
        h = m["norm2"](h)
        h = m["act"](h)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h0 = self.stem_forward(x, t_emb)              # (B,32,128,128)
        d1 = self.down_forward(self.down1, h0, t_emb) # (B,64,64,64)
        d2 = self.down_forward(self.down2, d1, t_emb) # (B,128,32,32)
        mid = self.mid_forward(d2, t_emb)             # (B,128,32,32)

        # Ups
        up2_in = torch.cat([mid, d2], dim=1)          # (B,256,32,32)
        u2 = self.up2["trans"](up2_in)                # (B,64,64,64)
        u2 = self.up2["block"](u2, t_emb)             # (B,64,64,64)

        up1_in = torch.cat([u2, d1], dim=1)           # (B,128,64,64)
        u1 = self.up1["trans"](up1_in)                # (B,32,128,128)
        u1 = self.up1["block"](u1, t_emb)             # (B,32,128,128)

        out = self.out(u1)                            # (B,1,128,128)
        return out


class TinyUNet(LegacyUNet):
    """Alias to keep the public name unchanged; matches legacy checkpoints."""
    pass


# --------------------------
# Dataset and degradation
# --------------------------


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, image_size: int = 128):
        self.paths = sorted(str(p) for p in Path(root).glob("*.tif"))
        if not self.paths:
            raise FileNotFoundError(f"No .tif patches found in {root}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.array(Image.open(self.paths[idx])).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.clip(arr, 0.0, 1.0)
        h, w = arr.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Patch {self.paths[idx]} has shape {arr.shape}, expected ({self.image_size},{self.image_size})")
        tensor = torch.from_numpy(arr)[None]  # (1,H,W)
        return tensor


@dataclass
class DegradeParams:
    lr_scale: int = 2
    blur_sigma: float = 0.8
    shot_noise_scale: float = 50.0
    gauss_std_min: float = 0.01
    gauss_std_max: float = 0.03


def degrade_to_lr(hr_01: torch.Tensor, params: DegradeParams) -> torch.Tensor:
    """
    hr_01: (N,1,H,W) in [0,1]
    returns lr_cond in [0,1] with same shape.
    """
    if params.blur_sigma > 0:
        kernel = make_gaussian_kernel2d(params.blur_sigma, hr_01.device)
        pad = kernel.shape[-1] // 2
        hr_01 = F.conv2d(hr_01, kernel.expand(hr_01.size(1), 1, -1, -1), padding=pad, groups=hr_01.size(1))

    # Downsample then upsample
    low = F.interpolate(hr_01, scale_factor=1.0 / params.lr_scale, mode="bilinear", align_corners=False, antialias=True)

    if params.shot_noise_scale > 0:
        low = torch.poisson(torch.clamp(low * params.shot_noise_scale, 0, None)) / params.shot_noise_scale

    if params.gauss_std_max > 0:
        sigma = torch.empty(low.shape[0], device=low.device).uniform_(params.gauss_std_min, params.gauss_std_max)
        sigma = sigma.view(-1, 1, 1, 1)
        low = low + torch.randn_like(low) * sigma

    low = torch.clamp(low, 0.0, 1.0)
    low_up = F.interpolate(low, scale_factor=params.lr_scale, mode="bilinear", align_corners=False)
    low_up = torch.clamp(low_up, 0.0, 1.0)
    return low_up


# --------------------------
# Training and sampling
# --------------------------


def prepare_batch(hr: torch.Tensor, params: DegradeParams, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    hr = hr.to(device)
    hr_01 = hr
    lr_01 = degrade_to_lr(hr_01, params)
    hr_m11 = hr_01 * 2.0 - 1.0
    lr_m11 = lr_01 * 2.0 - 1.0
    return hr_m11, lr_m11


def p_losses(model: TinyUNet, x_start: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    noise = torch.randn_like(x_start)
    alphas_cumprod = schedule.alphas_cumprod.to(x_start.device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]
    x_noisy = sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise
    model_in = torch.cat([x_noisy, cond], dim=1)
    pred = model(model_in, t)
    return F.mse_loss(pred, noise)


@torch.no_grad()
def sample_triplets(
    model: TinyUNet,
    schedule: DiffusionSchedule,
    hr_batch: torch.Tensor,
    params: DegradeParams,
    device: torch.device,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    hr_01 = hr_batch.to(device)
    lr_01 = degrade_to_lr(hr_01, params)
    hr = hr_01 * 2.0 - 1.0
    lr = lr_01 * 2.0 - 1.0

    x = torch.randn_like(hr)
    betas = schedule.betas.to(device)
    alphas = schedule.alphas.to(device)
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    for i in reversed(range(num_steps)):
        t = torch.full((hr.shape[0],), i, device=device, dtype=torch.long)
        model_in = torch.cat([x, lr], dim=1)
        eps = model(model_in, t)
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        beta = betas[i]
        coef1 = 1 / torch.sqrt(alpha)
        coef2 = beta / torch.sqrt(1 - alpha_bar)
        mean = coef1 * (x - coef2 * eps)
        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta) * noise
        else:
            x = mean

    sr = torch.clamp((x + 1) * 0.5, 0.0, 1.0)
    hr = torch.clamp((hr + 1) * 0.5, 0.0, 1.0)
    lr_up = torch.clamp((lr + 1) * 0.5, 0.0, 1.0)
    return lr_up.cpu(), sr.cpu(), hr.cpu()


def save_triplet_grid(lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor, path: Path, nrow: int = 3) -> None:
    lr_imgs = [img for img in lr]
    sr_imgs = [img for img in sr]
    hr_imgs = [img for img in hr]
    all_imgs: List[torch.Tensor] = []
    for trip in zip(lr_imgs, sr_imgs, hr_imgs):
        all_imgs.extend(trip)
    grid = make_grid(torch.stack(all_imgs, dim=0), nrow=nrow * 3, padding=2)
    save_image(grid, path)


# --------------------------
# Main
# --------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SR3 diffusion on MoS2 patches (conditional only).")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory with 128x128 .tif patches (float in [0,1]).")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to store checkpoints and samples.")
    p.add_argument("--run-name", type=str, default="mos2_sr3_fast", help="Subdirectory name under output-dir.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr-scale", type=int, default=2, help="Down/upsample factor for LR condition.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--num-timesteps", type=int, default=400)
    p.add_argument("--target-train-size", type=int, default=5000, help="Repeat dataset to reach this many samples per epoch.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blur-sigma", type=float, default=0.8)
    p.add_argument("--shot-noise-scale", type=float, default=50.0)
    p.add_argument("--gaussian-noise-std", type=float, default=0.01)
    p.add_argument("--gaussian-noise-std-max", type=float, default=0.03)
    p.add_argument("--sample-every", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = args.output_dir / args.run_name
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = PatchDataset(args.data_dir, image_size=args.image_size)
    base_len = len(dataset)
    repeat_factor = max(1, math.ceil(args.target_train_size / base_len))
    train_dataset = torch.utils.data.ConcatDataset([dataset] * repeat_factor)
    val_indices = random.sample(range(base_len), k=min(args.num_samples, base_len))
    val_hr = torch.stack([dataset[i] for i in val_indices], dim=0)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )

    params = DegradeParams(
        lr_scale=args.lr_scale,
        blur_sigma=args.blur_sigma,
        shot_noise_scale=args.shot_noise_scale,
        gauss_std_min=args.gaussian_noise_std,
        gauss_std_max=args.gaussian_noise_std_max,
    )
    schedule = make_beta_schedule(args.num_timesteps)

    model = TinyUNet(in_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Base patches: {base_len}")
    print(f"Repeat factor: {repeat_factor} (effective epoch size: {len(train_dataset)})")
    print(f"Model params (M): {sum(p.numel() for p in model.parameters()) / 1e6:.6f}")
    print(
        f"SR3 settings: lr_scale={args.lr_scale}, blur_sigma={args.blur_sigma}, "
        f"shot_noise_scale={args.shot_noise_scale}, gaussian_noise_std=[{args.gaussian_noise_std}, {args.gaussian_noise_std_max}]"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for hr_batch in dataloader:
            hr_m11, lr_m11 = prepare_batch(hr_batch, params, device)
            t = torch.randint(0, args.num_timesteps, (hr_batch.size(0),), device=device).long()
            loss = p_losses(model, hr_m11, lr_m11, t, schedule)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            lr_vis, sr_vis, hr_vis = sample_triplets(
                model, schedule, val_hr, params, device, num_steps=args.num_timesteps
            )
            save_path = sample_dir / f"epoch_{epoch:03d}.png"
            save_triplet_grid(lr_vis, sr_vis, hr_vis, save_path, nrow=args.num_samples)
            print(f"Saved samples to {save_path}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": {
                    "betas": schedule.betas,
                    "alphas": schedule.alphas,
                    "alphas_cumprod": schedule.alphas_cumprod,
                },
                "params": vars(args),
            }
            ckpt_path = out_dir / f"model_epoch_{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
