"""Generate multiple denoising GIFs from different partial maps.

Produces:
1. Multiple denoising GIFs (different floor plans, different coverages)
2. Side-by-side diversity GIF (8 completions denoising simultaneously)
3. Coverage sweep GIF (same map, increasing lidar range)
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import MapCompletionDataset
from diffusion import DDPMScheduler
from unet import ConditionalUNet

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL required for GIF generation")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ConditionalUNet(
        in_channels=3, out_channels=1,
        base_channels=args.get("base_channels", 32),
        channel_mults=tuple(args.get("channel_mults", [1, 2, 4, 4])),
        time_dim=args.get("time_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scheduler = DDPMScheduler(num_timesteps=args.get("T", 1000), device=device)
    return model, scheduler


def to_np(t):
    return (t[0, 0].cpu().numpy() + 1) / 2


def partial_rgb(p):
    vis = np.zeros((*p.shape, 3))
    vis[p > 0.6] = [1, 1, 1]
    vis[p < 0.4] = [0, 0, 0]
    vis[(p >= 0.4) & (p <= 0.6)] = [0.5, 0.5, 0.7]
    return vis


def iou(gt, pred):
    g, p = gt > 0.5, pred > 0.5
    return float((g & p).sum()) / max(float((g | p).sum()), 1)


def fig_to_frame(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    return Image.fromarray(np.asarray(buf).reshape(h, w, 4)[:, :, :3].copy())


@torch.no_grad()
def run_ddim_with_intermediates(model, scheduler, pm, km, device, num_steps=50):
    """Run DDIM and return intermediate x at each step."""
    x = torch.randn(1, 1, 256, 256, device=device)
    step_size = scheduler.T // num_steps
    timesteps = list(range(0, scheduler.T, step_size))[::-1]

    intermediates = [(scheduler.T, x.clone())]

    for i, t_val in enumerate(timesteps):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t, pm, km)
        ac = scheduler.alphas_cumprod[t_val]
        ac_prev = scheduler.alphas_cumprod[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0, device=device)
        pred_x0 = (x - torch.sqrt(1 - ac) * pred_noise) / torch.sqrt(ac)
        pred_x0 = pred_x0.clamp(-1, 1)
        x = torch.sqrt(ac_prev) * pred_x0 + torch.sqrt(1 - ac_prev) * pred_noise
        intermediates.append((t_val, x.clone()))

    return intermediates


def gif_1_single_denoise(model, scheduler, dataset, device, save_path, sample_idx=42):
    """Single denoising process with 4 panels."""
    sample = dataset[sample_idx]
    pm = sample["partial_map"].unsqueeze(0).to(device)
    km = sample["known_mask"].unsqueeze(0).to(device)
    fm = sample["full_map"].unsqueeze(0).to(device)
    gt = to_np(fm)
    partial = to_np(pm)

    intermediates = run_ddim_with_intermediates(model, scheduler, pm, km, device)
    pick = intermediates[::max(1, len(intermediates)//25)]
    if intermediates[-1] not in pick:
        pick.append(intermediates[-1])

    frames = []
    for t_val, x in pick:
        current = (x[0, 0].cpu().numpy() + 1) / 2
        progress = (1 - t_val / scheduler.T) * 100

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(partial_rgb(partial)); axes[0].set_title("Partial Map"); axes[0].axis("off")
        axes[1].imshow(current.clip(0,1), cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"Denoising: {progress:.0f}%"); axes[1].axis("off")
        axes[2].imshow(gt, cmap="gray_r", vmin=0, vmax=1); axes[2].set_title("Ground Truth"); axes[2].axis("off")
        axes[3].imshow(np.abs(gt - current.clip(0,1)), cmap="hot", vmin=0, vmax=1)
        axes[3].set_title(f"Error (IoU: {iou(gt, current):.2f})"); axes[3].axis("off")
        plt.suptitle("Diffusion Map Completion", fontsize=14, fontweight="bold")
        plt.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close()

    for _ in range(8):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"  Saved: {save_path} ({len(frames)} frames)")


@torch.no_grad()
def gif_2_diversity_denoise(model, scheduler, dataset, device, save_path, sample_idx=100, K=4):
    """K completions denoising side by side simultaneously."""
    sample = dataset[sample_idx]
    pm = sample["partial_map"].unsqueeze(0).to(device)
    km = sample["known_mask"].unsqueeze(0).to(device)
    fm = sample["full_map"].unsqueeze(0).to(device)
    gt = to_np(fm)
    partial = to_np(pm)

    all_intermediates = []
    for _ in range(K):
        all_intermediates.append(run_ddim_with_intermediates(model, scheduler, pm, km, device))

    n_frames = min(len(all_intermediates[0]), 26)
    step = max(1, len(all_intermediates[0]) // n_frames)

    frames = []
    for fi in range(0, len(all_intermediates[0]), step):
        fig, axes = plt.subplots(1, K + 2, figsize=(3.5 * (K + 2), 3.5))

        axes[0].imshow(partial_rgb(partial)); axes[0].set_title("Input"); axes[0].axis("off")

        t_val = all_intermediates[0][fi][0]
        progress = (1 - t_val / scheduler.T) * 100

        for k in range(K):
            current = (all_intermediates[k][fi][1][0, 0].cpu().numpy() + 1) / 2
            axes[k+1].imshow(current.clip(0,1), cmap="gray_r", vmin=0, vmax=1)
            axes[k+1].set_title(f"Sample {k+1}")
            axes[k+1].axis("off")

        axes[K+1].imshow(gt, cmap="gray_r", vmin=0, vmax=1); axes[K+1].set_title("GT"); axes[K+1].axis("off")

        plt.suptitle(f"Diverse Denoising: {progress:.0f}% complete", fontsize=14, fontweight="bold")
        plt.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close()

    for _ in range(8):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=250, loop=0)
    print(f"  Saved: {save_path} ({len(frames)} frames)")


@torch.no_grad()
def gif_3_zoom_denoise(model, scheduler, dataset, device, save_path, sample_idx=200):
    """Focused view: large prediction with zoomed-in detail panel."""
    sample = dataset[sample_idx]
    pm = sample["partial_map"].unsqueeze(0).to(device)
    km = sample["known_mask"].unsqueeze(0).to(device)
    fm = sample["full_map"].unsqueeze(0).to(device)
    gt = to_np(fm)
    partial = to_np(pm)

    intermediates = run_ddim_with_intermediates(model, scheduler, pm, km, device)
    pick = intermediates[::max(1, len(intermediates)//20)]
    if intermediates[-1] not in pick:
        pick.append(intermediates[-1])

    frames = []
    for t_val, x in pick:
        current = (x[0, 0].cpu().numpy() + 1) / 2
        progress = (1 - t_val / scheduler.T) * 100

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(partial_rgb(partial))
        axes[0].set_title("Partial Map (input)", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(current.clip(0,1), cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"Predicted ({progress:.0f}% denoised)", fontsize=12)
        axes[1].axis("off")

        diff = np.zeros((*gt.shape, 3))
        pred_bin = current.clip(0,1) > 0.5
        gt_bin = gt > 0.5
        known = km[0, 0].cpu().numpy()
        diff[gt_bin & pred_bin] = [0.9, 0.9, 0.9]
        diff[gt_bin & ~pred_bin & (known < 0.5)] = [0.3, 0.3, 0.9]
        diff[~gt_bin & pred_bin & (known < 0.5)] = [0.9, 0.3, 0.3]
        diff[~gt_bin & ~pred_bin] = [0.2, 0.2, 0.2]
        diff[known > 0.5] = [0.7, 0.7, 0.7]
        axes[2].imshow(diff)
        axes[2].set_title(f"Accuracy: IoU {iou(gt, current):.2f}\n(blue=missed, red=false)", fontsize=11)
        axes[2].axis("off")

        plt.suptitle("Map Completion Progress", fontsize=14, fontweight="bold")
        plt.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close()

    for _ in range(8):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"  Saved: {save_path} ({len(frames)} frames)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/gifs")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)
    print("Loading dataset...")
    dataset = MapCompletionDataset(args.data_dir)
    print(f"  {len(dataset)} samples\n")

    sample_indices = [42, 100, 200, 500, 800, 1000, 1200, 1500]

    print("=== Single denoising GIFs (8 different maps) ===")
    for i, idx in enumerate(sample_indices):
        if idx < len(dataset):
            gif_1_single_denoise(model, scheduler, dataset, device,
                                  os.path.join(args.out_dir, f"denoise_{i+1:02d}.gif"),
                                  sample_idx=idx)

    print("\n=== Diversity denoising GIF (4 samples side by side) ===")
    gif_2_diversity_denoise(model, scheduler, dataset, device,
                             os.path.join(args.out_dir, "diversity_denoise.gif"),
                             sample_idx=300)

    print("\n=== Accuracy tracking GIFs (3 maps) ===")
    for i, idx in enumerate([150, 400, 700]):
        if idx < len(dataset):
            gif_3_zoom_denoise(model, scheduler, dataset, device,
                                os.path.join(args.out_dir, f"accuracy_{i+1:02d}.gif"),
                                sample_idx=idx)

    print(f"\nAll GIFs saved to {args.out_dir}/")
