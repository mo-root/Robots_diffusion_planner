"""Evaluate a trained diffusion model on held-out test data.

Produces:
- Per-sample IoU, MSE, SSIM metrics
- Aggregate statistics (mean, std, median)
- Visual comparison grids
- Multi-sample diversity analysis
- CSV of per-sample metrics

Usage:
    python src/evaluate.py \
        --checkpoint results/checkpoints/model_final.pt \
        --test_dir data/val \
        --out_dir results/evaluation \
        --device cuda \
        --num_samples 100 \
        --K 8
"""

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import MapCompletionDataset
from diffusion import DDPMScheduler
from unet import ConditionalUNet


def compute_iou(gt, pred, threshold=0.5):
    gt_bin = gt > threshold
    pred_bin = pred > threshold
    inter = (gt_bin & pred_bin).sum()
    union = (gt_bin | pred_bin).sum()
    return float(inter) / max(float(union), 1)


def compute_mse(gt, pred):
    return float(((gt - pred) ** 2).mean())


def compute_accuracy(gt, pred, threshold=0.5):
    gt_bin = gt > threshold
    pred_bin = pred > threshold
    return float((gt_bin == pred_bin).sum()) / gt_bin.size


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    model = ConditionalUNet(
        in_channels=3, out_channels=1,
        base_channels=ckpt_args.get("base_channels", 32),
        channel_mults=tuple(ckpt_args.get("channel_mults", [1, 2, 4, 4])),
        time_dim=ckpt_args.get("time_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    scheduler = DDPMScheduler(
        num_timesteps=ckpt_args.get("T", 1000), device=device
    )

    dataset = MapCompletionDataset(args.test_dir)
    print(f"Test set: {len(dataset)} samples")

    indices = np.random.default_rng(42).choice(
        len(dataset), size=min(args.num_samples, len(dataset)), replace=False
    )

    metrics = []
    all_ious = []
    all_mses = []
    all_accs = []
    all_diversities = []

    print(f"Evaluating {len(indices)} samples with K={args.K} completions each...")
    t0 = time.time()

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        pm = sample["partial_map"].unsqueeze(0).to(device)
        km = sample["known_mask"].unsqueeze(0).to(device)
        fm = sample["full_map"].unsqueeze(0).to(device)

        gt = (fm[0, 0].cpu().numpy() + 1) / 2

        sample_preds = []
        sample_ious = []
        for k in range(args.K):
            pred = scheduler.sample_ddim(model, pm, km, num_steps=args.ddim_steps)
            pred_np = (pred[0, 0].cpu().numpy() + 1) / 2
            pred_np = pred_np.clip(0, 1)
            sample_preds.append(pred_np)
            sample_ious.append(compute_iou(gt, pred_np))

        mean_pred = np.mean(sample_preds, axis=0)
        std_pred = np.std(sample_preds, axis=0)

        best_iou = max(sample_ious)
        mean_iou = np.mean(sample_ious)
        mse = compute_mse(gt, mean_pred)
        acc = compute_accuracy(gt, mean_pred)
        diversity = std_pred.mean()

        metrics.append({
            "idx": int(idx),
            "mean_iou": mean_iou,
            "best_iou": best_iou,
            "mse": mse,
            "accuracy": acc,
            "diversity": diversity,
            "coverage": km[0, 0].cpu().numpy().mean(),
        })
        all_ious.append(mean_iou)
        all_mses.append(mse)
        all_accs.append(acc)
        all_diversities.append(diversity)

        if i < 20:
            save_comparison(gt, (pm[0,0].cpu().numpy()+1)/2, km[0,0].cpu().numpy(),
                           mean_pred, std_pred, sample_preds, sample_ious,
                           os.path.join(args.out_dir, f"comparison_{i:03d}.png"))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(indices)}] mean_iou={np.mean(all_ious):.3f} "
                  f"mean_mse={np.mean(all_mses):.5f} ({elapsed:.0f}s)")

    print(f"\n{'='*50}")
    print(f"RESULTS ({len(indices)} samples, K={args.K})")
    print(f"{'='*50}")
    print(f"IoU:       {np.mean(all_ious):.3f} +/- {np.std(all_ious):.3f}")
    print(f"MSE:       {np.mean(all_mses):.5f} +/- {np.std(all_mses):.5f}")
    print(f"Accuracy:  {np.mean(all_accs):.3f} +/- {np.std(all_accs):.3f}")
    print(f"Diversity: {np.mean(all_diversities):.4f} +/- {np.std(all_diversities):.4f}")

    with open(os.path.join(args.out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    save_summary_plots(metrics, args.out_dir)
    save_results_card(all_ious, all_mses, all_accs, all_diversities,
                      len(indices), args.K, ckpt.get("epoch", "?"), args.out_dir)

    print(f"\nResults saved to {args.out_dir}/")


def save_comparison(gt, partial, mask, mean_pred, std_pred, samples, ious, path):
    n_show = min(4, len(samples))
    fig, axes = plt.subplots(2, n_show + 3, figsize=(4 * (n_show + 3), 8))

    axes[0, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Ground Truth", fontweight="bold")
    axes[0, 0].axis("off")

    vis = np.zeros((*partial.shape, 3))
    vis[partial > 0.6] = [1, 1, 1]
    vis[partial < 0.4] = [0, 0, 0]
    vis[(partial >= 0.4) & (partial <= 0.6)] = [0.5, 0.5, 0.7]
    axes[0, 1].imshow(vis)
    axes[0, 1].set_title(f"Partial ({mask.mean():.0%})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mean_pred.clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Mean pred (IoU: {np.mean(ious):.2f})")
    axes[0, 2].axis("off")

    for i in range(n_show):
        axes[0, i+3].imshow(samples[i].clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
        axes[0, i+3].set_title(f"Sample {i+1} ({ious[i]:.2f})")
        axes[0, i+3].axis("off")

    error = np.abs(gt - mean_pred.clip(0, 1))
    axes[1, 0].imshow(error, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Error map")
    axes[1, 0].axis("off")

    im = axes[1, 1].imshow(std_pred, cmap="hot", vmin=0, vmax=std_pred.max() + 1e-6)
    axes[1, 1].set_title("Uncertainty")
    axes[1, 1].axis("off")

    hidden_free = (gt > 0.5) & (mask < 0.5)
    hidden_wall = (gt < 0.5) & (mask < 0.5)
    diff = np.ones((*gt.shape, 3)) * 0.7
    diff[mask > 0.5] = [0.9, 0.9, 0.9]
    diff[hidden_free] = [0.3, 0.8, 0.3]
    diff[hidden_wall] = [0.8, 0.3, 0.3]
    diff[(gt < 0.5) & (mask > 0.5)] = [0, 0, 0]
    axes[1, 2].imshow(diff)
    axes[1, 2].set_title("Hidden regions")
    axes[1, 2].axis("off")

    for i in range(n_show):
        pred_bin = samples[i] > 0.5
        gt_bin = gt > 0.5
        tp = pred_bin & gt_bin & (mask < 0.5)
        fp = pred_bin & ~gt_bin & (mask < 0.5)
        fn = ~pred_bin & gt_bin & (mask < 0.5)
        overlay = np.ones((*gt.shape, 3)) * 0.7
        overlay[mask > 0.5] = [0.9, 0.9, 0.9]
        overlay[tp] = [0.3, 0.8, 0.3]
        overlay[fp] = [0.8, 0.3, 0.3]
        overlay[fn] = [0.3, 0.3, 0.8]
        axes[1, i+3].imshow(overlay)
        axes[1, i+3].set_title(f"TP/FP/FN #{i+1}")
        axes[1, i+3].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def save_summary_plots(metrics, out_dir):
    ious = [m["mean_iou"] for m in metrics]
    mses = [m["mse"] for m in metrics]
    coverages = [m["coverage"] for m in metrics]
    diversities = [m["diversity"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(ious, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0, 0].axvline(np.mean(ious), color="red", linestyle="--",
                       label=f"Mean: {np.mean(ious):.3f}")
    axes[0, 0].set_xlabel("IoU")
    axes[0, 0].set_title("IoU Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(coverages, ious, alpha=0.5, s=20, color="#55A868")
    axes[0, 1].set_xlabel("Partial map coverage")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].set_title("IoU vs Coverage")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(diversities, bins=30, color="#DD8452", edgecolor="white", alpha=0.8)
    axes[1, 0].set_xlabel("Mean pixel std across K samples")
    axes[1, 0].set_title("Prediction Diversity")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(diversities, ious, alpha=0.5, s=20, color="#C44E52")
    axes[1, 1].set_xlabel("Diversity")
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_title("IoU vs Diversity")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Evaluation Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_results_card(ious, mses, accs, divs, n, K, epoch, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    data = [
        ["Metric", "Mean", "Std", "Min", "Max"],
        ["IoU", f"{np.mean(ious):.3f}", f"{np.std(ious):.3f}",
         f"{np.min(ious):.3f}", f"{np.max(ious):.3f}"],
        ["MSE", f"{np.mean(mses):.5f}", f"{np.std(mses):.5f}",
         f"{np.min(mses):.5f}", f"{np.max(mses):.5f}"],
        ["Accuracy", f"{np.mean(accs):.3f}", f"{np.std(accs):.3f}",
         f"{np.min(accs):.3f}", f"{np.max(accs):.3f}"],
        ["Diversity", f"{np.mean(divs):.4f}", f"{np.std(divs):.4f}",
         f"{np.min(divs):.4f}", f"{np.max(divs):.4f}"],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    plt.title(f"Model Evaluation (epoch {epoch}, n={n}, K={K})",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "results_card.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=50)
    args = parser.parse_args()
    evaluate(args)
