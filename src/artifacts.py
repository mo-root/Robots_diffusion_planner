"""Generate visual artifacts for project showcase.

Produces:
- Denoising process GIFs (noise -> predicted map, step by step)
- Multi-sample diversity grids (K completions from same partial map)
- Before/after comparison strips
- Training progress montages
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from PIL import Image
except ImportError:
    Image = None


def save_denoising_gif(model, scheduler, partial_map, known_mask, device,
                       save_path, num_steps=50, fps=10):
    """Create a GIF showing the reverse diffusion process step by step."""
    if Image is None:
        print("PIL not available, skipping GIF generation")
        return

    model.eval()
    b = partial_map.shape[0]
    x = torch.randn(b, 1, partial_map.shape[2], partial_map.shape[3], device=device)

    step_size = scheduler.T // num_steps
    timesteps = list(range(0, scheduler.T, step_size))[::-1]

    frames = []
    frame_every = max(1, len(timesteps) // 30)

    for i, t_val in enumerate(timesteps):
        t = torch.full((b,), t_val, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t, partial_map, known_mask)

        alpha_cumprod_t = scheduler.alphas_cumprod[t_val]
        alpha_cumprod_prev = (
            scheduler.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else
            torch.tensor(1.0, device=device)
        )

        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = pred_x0.clamp(-1, 1)

        dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * pred_noise
        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

        if i % frame_every == 0 or i == len(timesteps) - 1:
            img = (x[0, 0].cpu().numpy() + 1) / 2
            img = (img.clip(0, 1) * 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            pm = (partial_map[0, 0].cpu().numpy() + 1) / 2
            vis = np.zeros((*pm.shape, 3), dtype=np.uint8)
            vis[pm > 0.6] = [255, 255, 255]
            vis[pm < 0.4] = [0, 0, 0]
            vis[(pm >= 0.4) & (pm <= 0.6)] = [128, 128, 180]
            axes[0].imshow(vis)
            axes[0].set_title("Partial Map (input)")
            axes[0].axis("off")

            axes[1].imshow(img, cmap="gray_r", vmin=0, vmax=255)
            axes[1].set_title(f"Denoising step {i+1}/{len(timesteps)} (t={t_val})")
            axes[1].axis("off")

            km = known_mask[0, 0].cpu().numpy()
            axes[2].imshow(img, cmap="gray_r", vmin=0, vmax=255)
            axes[2].set_title(f"Progress: {(1 - t_val/scheduler.T)*100:.0f}%")
            axes[2].axis("off")

            plt.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            frame_arr = np.asarray(buf).reshape(h, w, 4)[:, :, :3].copy()
            frames.append(Image.fromarray(frame_arr))
            plt.close()

    for _ in range(5):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=1000 // fps, loop=0)
    print(f"Denoising GIF saved: {save_path} ({len(frames)} frames)")


def save_diversity_grid(model, scheduler, partial_map, known_mask, full_map,
                        device, save_path, k=8, num_steps=50):
    """Show K different completions from the same partial map."""
    model.eval()

    samples = []
    for _ in range(k):
        pred = scheduler.sample_ddim(model, partial_map[:1], known_mask[:1], num_steps=num_steps)
        samples.append((pred[0, 0].cpu().numpy() + 1) / 2)

    fig, axes = plt.subplots(2, (k + 2) // 2 + 1, figsize=(4 * ((k + 2) // 2 + 1), 8))
    axes = axes.flatten()

    gt = (full_map[0, 0].cpu().numpy() + 1) / 2
    axes[0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].axis("off")

    pm = (partial_map[0, 0].cpu().numpy() + 1) / 2
    vis = np.zeros((*pm.shape, 3))
    vis[pm > 0.6] = [1, 1, 1]
    vis[pm < 0.4] = [0, 0, 0]
    vis[(pm >= 0.4) & (pm <= 0.6)] = [0.5, 0.5, 0.7]
    axes[1].imshow(vis)
    axes[1].set_title("Partial Map", fontweight="bold")
    axes[1].axis("off")

    for i, s in enumerate(samples):
        axes[i + 2].imshow(s.clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
        iou = compute_iou(gt, s)
        axes[i + 2].set_title(f"Sample {i+1} (IoU: {iou:.2f})")
        axes[i + 2].axis("off")

    for j in range(len(samples) + 2, len(axes)):
        axes[j].axis("off")

    mean_pred = np.mean(samples, axis=0)
    std_pred = np.std(samples, axis=0)

    plt.suptitle(f"Diversity: {k} samples from same partial map\n"
                 f"Mean IoU: {np.mean([compute_iou(gt, s) for s in samples]):.3f} | "
                 f"Avg pixel std: {std_pred.mean():.3f}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diversity grid saved: {save_path}")


def save_uncertainty_map(model, scheduler, partial_map, known_mask,
                         device, save_path, k=8, num_steps=50):
    """Visualize where the model is most uncertain (high variance across samples)."""
    model.eval()
    samples = []
    for _ in range(k):
        pred = scheduler.sample_ddim(model, partial_map[:1], known_mask[:1], num_steps=num_steps)
        samples.append((pred[0, 0].cpu().numpy() + 1) / 2)

    mean_pred = np.mean(samples, axis=0)
    std_pred = np.std(samples, axis=0)
    km = known_mask[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pm = (partial_map[0, 0].cpu().numpy() + 1) / 2
    vis = np.zeros((*pm.shape, 3))
    vis[pm > 0.6] = [1, 1, 1]
    vis[pm < 0.4] = [0, 0, 0]
    vis[(pm >= 0.4) & (pm <= 0.6)] = [0.5, 0.5, 0.7]
    axes[0].imshow(vis)
    axes[0].set_title("Partial Map")
    axes[0].axis("off")

    axes[1].imshow(mean_pred.clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
    axes[1].set_title("Mean Prediction")
    axes[1].axis("off")

    im = axes[2].imshow(std_pred, cmap="hot", vmin=0, vmax=std_pred.max())
    axes[2].set_title("Uncertainty (std across samples)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Uncertainty map saved: {save_path}")


def save_training_montage(log_dir: str, save_path: str):
    """Combine all epoch sample images into a single training progress montage."""
    if Image is None:
        return

    sample_files = sorted(Path(log_dir).glob("samples_epoch*.png"))
    if len(sample_files) < 2:
        return

    pick = sample_files[::max(1, len(sample_files) // 8)]
    if sample_files[-1] not in pick:
        pick.append(sample_files[-1])

    images = [Image.open(f) for f in pick]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    montage = Image.new("RGB", (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        montage.paste(img, (x_offset, 0))
        x_offset += img.width

    montage.save(save_path)
    print(f"Training montage saved: {save_path}")


def compute_iou(gt, pred, threshold=0.5):
    gt_bin = gt > threshold
    pred_bin = pred > threshold
    intersection = (gt_bin & pred_bin).sum()
    union = (gt_bin | pred_bin).sum()
    return intersection / max(union, 1)
