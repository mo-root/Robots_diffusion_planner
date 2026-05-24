"""Generate comprehensive showcase artifacts from a trained model.

Produces:
1. Large comparison grid (GT vs Partial vs Predicted vs Error) - 12 samples
2. Denoising process GIF (noise -> map step by step)
3. Diversity showcase (K=8 completions from 4 different partial maps)
4. Uncertainty heatmaps
5. Best & worst predictions side by side
6. Training progression montage (epoch 5 vs 10 vs 15 vs 20 vs 25)
7. IoU distribution histogram from evaluation
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import MapCompletionDataset
from diffusion import DDPMScheduler
from unet import ConditionalUNet

try:
    from PIL import Image
except ImportError:
    Image = None


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
    return model, scheduler, ckpt.get("epoch", "?")


def to_numpy(tensor):
    return (tensor[0, 0].cpu().numpy() + 1) / 2


def partial_to_rgb(partial_np):
    vis = np.zeros((*partial_np.shape, 3))
    vis[partial_np > 0.6] = [1, 1, 1]
    vis[partial_np < 0.4] = [0, 0, 0]
    vis[(partial_np >= 0.4) & (partial_np <= 0.6)] = [0.5, 0.5, 0.7]
    return vis


def compute_iou(gt, pred, threshold=0.5):
    gt_bin = gt > threshold
    pred_bin = pred > threshold
    inter = (gt_bin & pred_bin).sum()
    union = (gt_bin | pred_bin).sum()
    return float(inter) / max(float(union), 1)


@torch.no_grad()
def showcase_1_comparison_grid(model, scheduler, dataset, device, save_path, n=12):
    """Large grid: GT | Partial | Predicted | Error for n samples."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(16, 3.5 * n))

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        pm = sample["partial_map"].unsqueeze(0).to(device)
        km = sample["known_mask"].unsqueeze(0).to(device)
        fm = sample["full_map"].unsqueeze(0).to(device)

        pred = scheduler.sample_ddim(model, pm, km, num_steps=50)

        gt = to_numpy(fm)
        partial = to_numpy(pm)
        known = km[0, 0].cpu().numpy()
        predicted = to_numpy(pred).clip(0, 1)
        iou = compute_iou(gt, predicted)

        axes[row, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("Ground Truth" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_to_rgb(partial))
        axes[row, 1].set_title(f"Partial ({known.mean():.0%})" if row == 0 else f"({known.mean():.0%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted" if row == 0 else "")
        axes[row, 2].axis("off")

        error = np.abs(gt - predicted)
        axes[row, 3].imshow(error, cmap="hot", vmin=0, vmax=1)
        axes[row, 3].set_title(f"Error (IoU: {iou:.2f})" if row == 0 else f"IoU: {iou:.2f}")
        axes[row, 3].axis("off")

    plt.suptitle("Model Predictions: Ground Truth vs Partial Input vs Prediction vs Error",
                 fontsize=16, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[1/6] Comparison grid saved: {save_path}")


@torch.no_grad()
def showcase_2_denoising_gif(model, scheduler, dataset, device, save_path, num_steps=50):
    """GIF showing reverse diffusion: noise -> predicted map."""
    if Image is None:
        print("[2/6] Skipped (PIL not available)")
        return

    sample = dataset[42]
    pm = sample["partial_map"].unsqueeze(0).to(device)
    km = sample["known_mask"].unsqueeze(0).to(device)
    fm = sample["full_map"].unsqueeze(0).to(device)

    gt = to_numpy(fm)
    partial = to_numpy(pm)
    known = km[0, 0].cpu().numpy()

    x = torch.randn(1, 1, 256, 256, device=device)
    step_size = scheduler.T // num_steps
    timesteps = list(range(0, scheduler.T, step_size))[::-1]

    frames = []
    frame_every = max(1, len(timesteps) // 25)

    for i, t_val in enumerate(timesteps):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t, pm, km)

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
            current = (x[0, 0].cpu().numpy() + 1) / 2
            progress = (1 - t_val / scheduler.T) * 100

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(partial_to_rgb(partial))
            axes[0].set_title("Input: Partial Map")
            axes[0].axis("off")

            axes[1].imshow(current.clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
            axes[1].set_title(f"Denoising: {progress:.0f}% (t={t_val})")
            axes[1].axis("off")

            axes[2].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

            iou = compute_iou(gt, current.clip(0, 1))
            axes[3].imshow(np.abs(gt - current.clip(0, 1)), cmap="hot", vmin=0, vmax=1)
            axes[3].set_title(f"Error (IoU: {iou:.2f})")
            axes[3].axis("off")

            plt.suptitle(f"Reverse Diffusion Process", fontsize=14, fontweight="bold")
            plt.tight_layout()

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            frame_arr = np.asarray(buf).reshape(h, w, 4)[:, :, :3].copy()
            frames.append(Image.fromarray(frame_arr))
            plt.close()

    for _ in range(8):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                   duration=200, loop=0)
    print(f"[2/6] Denoising GIF saved: {save_path} ({len(frames)} frames)")


@torch.no_grad()
def showcase_3_diversity(model, scheduler, dataset, device, save_path, K=8, n_maps=4):
    """Show K completions from n different partial maps."""
    rng = np.random.default_rng(99)
    indices = rng.choice(len(dataset), size=n_maps, replace=False)

    fig, axes = plt.subplots(n_maps, K + 2, figsize=(3 * (K + 2), 3.5 * n_maps))

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        pm = sample["partial_map"].unsqueeze(0).to(device)
        km = sample["known_mask"].unsqueeze(0).to(device)
        fm = sample["full_map"].unsqueeze(0).to(device)

        gt = to_numpy(fm)
        partial = to_numpy(pm)
        known = km[0, 0].cpu().numpy()

        axes[row, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("GT" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_to_rgb(partial))
        axes[row, 1].set_title(f"Partial ({known.mean():.0%})" if row == 0 else f"({known.mean():.0%})")
        axes[row, 1].axis("off")

        ious = []
        for k in range(K):
            pred = scheduler.sample_ddim(model, pm, km, num_steps=50)
            pred_np = to_numpy(pred).clip(0, 1)
            iou = compute_iou(gt, pred_np)
            ious.append(iou)

            axes[row, k + 2].imshow(pred_np, cmap="gray_r", vmin=0, vmax=1)
            axes[row, k + 2].set_title(f"S{k+1} ({iou:.2f})" if row == 0 else f"{iou:.2f}")
            axes[row, k + 2].axis("off")

    plt.suptitle(f"Sample Diversity: {K} completions per partial map\n"
                 f"Each column is a different random sample from the same model",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[3/6] Diversity grid saved: {save_path}")


@torch.no_grad()
def showcase_4_uncertainty(model, scheduler, dataset, device, save_path, K=8, n_maps=4):
    """Uncertainty heatmaps showing where model disagrees across samples."""
    rng = np.random.default_rng(77)
    indices = rng.choice(len(dataset), size=n_maps, replace=False)

    fig, axes = plt.subplots(n_maps, 4, figsize=(16, 4 * n_maps))

    for row, idx in enumerate(indices):
        sample = dataset[int(idx)]
        pm = sample["partial_map"].unsqueeze(0).to(device)
        km = sample["known_mask"].unsqueeze(0).to(device)
        fm = sample["full_map"].unsqueeze(0).to(device)

        gt = to_numpy(fm)
        partial = to_numpy(pm)
        known = km[0, 0].cpu().numpy()

        preds = []
        for _ in range(K):
            pred = scheduler.sample_ddim(model, pm, km, num_steps=50)
            preds.append(to_numpy(pred).clip(0, 1))

        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)

        axes[row, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("Ground Truth" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_to_rgb(partial))
        axes[row, 1].set_title(f"Partial ({known.mean():.0%})" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1)
        iou = compute_iou(gt, mean_pred)
        axes[row, 2].set_title(f"Mean (IoU: {iou:.2f})" if row == 0 else f"IoU: {iou:.2f}")
        axes[row, 2].axis("off")

        im = axes[row, 3].imshow(std_pred, cmap="hot", vmin=0, vmax=max(std_pred.max(), 0.01))
        axes[row, 3].set_title("Uncertainty" if row == 0 else "")
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    plt.suptitle(f"Model Uncertainty: bright = high disagreement across {K} samples\n"
                 f"These uncertain regions are where frontier exploration adds most value",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[4/6] Uncertainty maps saved: {save_path}")


@torch.no_grad()
def showcase_5_best_worst(model, scheduler, dataset, device, save_path, n_eval=50):
    """Find and display the best and worst predictions."""
    rng = np.random.default_rng(123)
    indices = rng.choice(len(dataset), size=min(n_eval, len(dataset)), replace=False)

    results = []
    for idx in indices:
        sample = dataset[int(idx)]
        pm = sample["partial_map"].unsqueeze(0).to(device)
        km = sample["known_mask"].unsqueeze(0).to(device)
        fm = sample["full_map"].unsqueeze(0).to(device)

        pred = scheduler.sample_ddim(model, pm, km, num_steps=50)

        gt = to_numpy(fm)
        predicted = to_numpy(pred).clip(0, 1)
        partial = to_numpy(pm)
        known = km[0, 0].cpu().numpy()
        iou = compute_iou(gt, predicted)

        results.append((iou, gt, partial, known, predicted))

    results.sort(key=lambda x: x[0], reverse=True)
    best_5 = results[:5]
    worst_5 = results[-5:]

    fig, axes = plt.subplots(10, 4, figsize=(16, 35))

    for i, (iou, gt, partial, known, predicted) in enumerate(best_5):
        r = i
        axes[r, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"Best #{i+1}", fontsize=11, fontweight="bold", color="green")
        axes[r, 0].set_title("GT" if i == 0 else "")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(partial_to_rgb(partial))
        axes[r, 1].set_title(f"Partial ({known.mean():.0%})" if i == 0 else "")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[r, 2].set_title("Predicted" if i == 0 else "")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(np.abs(gt - predicted), cmap="hot", vmin=0, vmax=1)
        axes[r, 3].set_title(f"IoU: {iou:.3f}" if i == 0 else f"{iou:.3f}")
        axes[r, 3].axis("off")

    for i, (iou, gt, partial, known, predicted) in enumerate(worst_5):
        r = i + 5
        axes[r, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(f"Worst #{i+1}", fontsize=11, fontweight="bold", color="red")
        axes[r, 0].set_title("")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(partial_to_rgb(partial))
        axes[r, 1].axis("off")

        axes[r, 2].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[r, 2].axis("off")

        axes[r, 3].imshow(np.abs(gt - predicted), cmap="hot", vmin=0, vmax=1)
        axes[r, 3].set_title(f"{iou:.3f}")
        axes[r, 3].axis("off")

    plt.suptitle("Best 5 vs Worst 5 Predictions (out of 50 test samples)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[5/6] Best/worst predictions saved: {save_path}")


def showcase_6_training_progression(results_dir, save_path):
    """Side-by-side comparison of predictions at different training epochs."""
    epoch_files = sorted(Path(results_dir).glob("samples/samples_epoch*.png"))
    if not epoch_files:
        print("[6/6] Skipped (no epoch sample files found)")
        return

    if Image is None:
        print("[6/6] Skipped (PIL not available)")
        return

    images = [Image.open(f) for f in epoch_files]
    labels = [f.stem.replace("samples_epoch", "Epoch ").lstrip("0").replace("Epoch ", "Epoch ") for f in epoch_files]

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(np.array(img))
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Training Progression: How Predictions Improve Over Time",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[6/6] Training progression saved: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="results/showcase")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler, epoch = load_model(args.checkpoint, device)
    print(f"  Loaded epoch {epoch}")

    print("Loading dataset...")
    dataset = MapCompletionDataset(args.data_dir)
    print(f"  {len(dataset)} samples")

    print("\nGenerating showcase artifacts...\n")

    showcase_1_comparison_grid(model, scheduler, dataset, device,
                               os.path.join(args.out_dir, "01_comparison_grid.png"))

    showcase_2_denoising_gif(model, scheduler, dataset, device,
                              os.path.join(args.out_dir, "02_denoising_process.gif"))

    showcase_3_diversity(model, scheduler, dataset, device,
                          os.path.join(args.out_dir, "03_sample_diversity.png"))

    showcase_4_uncertainty(model, scheduler, dataset, device,
                            os.path.join(args.out_dir, "04_uncertainty_maps.png"))

    showcase_5_best_worst(model, scheduler, dataset, device,
                           os.path.join(args.out_dir, "05_best_worst.png"))

    showcase_6_training_progression(args.results_dir,
                                     os.path.join(args.out_dir, "06_training_progression.png"))

    print(f"\nAll showcase artifacts saved to {args.out_dir}/")
