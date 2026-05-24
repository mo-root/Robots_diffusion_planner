"""Generate presentation-ready artifacts for the project showcase.

Produces:
1. Polished training loss + IoU dual plot
2. Architecture pipeline diagram (ASCII -> matplotlib)
3. Frontier scoring worked example visualization
4. Results summary poster card
5. IoU vs coverage scatter with trend line
6. Per-epoch loss table as styled image
7. Before/after strip (single row, clean)
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def artifact_1_polished_loss_plot(history_path, save_path):
    """Publication-quality loss + IoU plot."""
    with open(history_path) as f:
        h = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = h["epoch"]
    train_loss = h["train_loss"]
    val_data = [(e, v) for e, v in zip(epochs, h["val_loss"]) if v is not None]
    iou_data = [(e, v) for e, v in zip(epochs, h.get("iou", [])) if v is not None]

    ax1.semilogy(epochs, train_loss, color="#2196F3", linewidth=2, label="Train Loss")
    if val_data:
        ve, vl = zip(*val_data)
        ax1.semilogy(ve, vl, color="#FF9800", linewidth=2, label="Val Loss", marker="o", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss (log scale)", fontsize=12)
    ax1.set_title("Training Convergence", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(epochs))

    if iou_data:
        ie, iv = zip(*iou_data)
        ax2.plot(ie, iv, color="#4CAF50", linewidth=2, marker="o", markersize=5)
        ax2.fill_between(ie, iv, alpha=0.1, color="#4CAF50")
        ax2.axhline(y=np.mean(iv), color="#4CAF50", linestyle="--", alpha=0.5,
                     label=f"Mean: {np.mean(iv):.3f}")
        ax2.set_ylim(0, 1)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation IoU", fontsize=12)
    ax2.set_title("Map Prediction Quality", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(epochs))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[1/7] Polished loss plot saved: {save_path}")


def artifact_2_pipeline_diagram(save_path):
    """Visual pipeline diagram: data -> model -> frontier scoring."""
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1, 2.5, "HouseExpo\n35k floor plans", "#E3F2FD", "#1565C0"),
        (4, 2.5, "Data Pipeline\nLidar sim + augment\n2.7M pairs", "#E8F5E9", "#2E7D32"),
        (7, 2.5, "U-Net (4.3M)\nDDPM Training\n29 epochs", "#FFF3E0", "#E65100"),
        (10, 2.5, "DDIM Sampling\nK=8 completions\n50 steps", "#F3E5F5", "#6A1B9A"),
        (13, 2.5, "Frontier Scorer\nE[gain] + λ·Std\n- β·dist", "#FFEBEE", "#B71C1C"),
        (16, 2.5, "Robot Action\nDrive to best\nfrontier", "#E0F7FA", "#006064"),
    ]

    for x, y, text, facecolor, edgecolor in boxes:
        box = FancyBboxPatch((x - 1.2, y - 0.9), 2.4, 1.8,
                             boxstyle="round,pad=0.15", facecolor=facecolor,
                             edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                fontweight="bold", color=edgecolor)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.3
        x2 = boxes[i+1][0] - 1.3
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="-|>", color="#424242", lw=2))

    ax.text(9, 5.2, "Diffusion-Based Map Completion for Frontier Exploration",
            ha="center", fontsize=16, fontweight="bold", color="#212121")
    ax.text(9, 4.6, "COSC 81/281 Final Project — Moin Mattar",
            ha="center", fontsize=12, color="#616161")

    phase_labels = [
        (2.5, 0.5, "Phase 1: Data", "#1565C0"),
        (7, 0.5, "Phase 1: Training", "#E65100"),
        (11.5, 0.5, "Phase 2: Inference", "#6A1B9A"),
        (14.5, 0.5, "Phase 2: Integration", "#B71C1C"),
    ]
    for x, y, text, color in phase_labels:
        ax.text(x, y, text, ha="center", fontsize=10, color=color, style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[2/7] Pipeline diagram saved: {save_path}")


def artifact_3_frontier_scoring_example(save_path):
    """Visual explanation of how frontier scoring works."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    np.random.seed(42)
    grid_size = 20

    partial = np.ones((grid_size, grid_size)) * 0.5
    partial[5:15, 5:12] = 1.0
    partial[5, 5:12] = 0.0
    partial[14, 5:12] = 0.0
    partial[5:15, 5] = 0.0
    partial[5:15, 11] = 0.0
    partial[8:12, 8:10] = 0.0

    vis = np.zeros((*partial.shape, 3))
    vis[partial > 0.7] = [1, 1, 1]
    vis[partial < 0.3] = [0, 0, 0]
    vis[(partial > 0.3) & (partial < 0.7)] = [0.6, 0.6, 0.8]
    axes[0].imshow(vis, interpolation="nearest")
    axes[0].set_title("Step 1: Partial Map\n(from PA4 mapper)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    comp1 = partial.copy()
    comp1[comp1 == 0.5] = np.random.choice([0, 1], size=(comp1 == 0.5).sum(), p=[0.3, 0.7])
    comp1_vis = np.stack([comp1]*3, axis=-1)
    comp1_vis[partial == 0.5] = [0.8, 1.0, 0.8]
    comp1_vis[partial < 0.3] = [0, 0, 0]
    axes[1].imshow(comp1_vis, interpolation="nearest")
    axes[1].set_title("Step 2: Sample K=8\nmap completions", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    frontier_y = [5, 5, 14, 14, 10, 10]
    frontier_x = [4, 12, 4, 12, 4, 12]
    scores = [0.8, 0.3, 0.6, 0.9, 0.4, 0.7]

    score_vis = vis.copy()
    for fy, fx, s in zip(frontier_y, frontier_x, scores):
        color = [s, 1-s, 0]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = fy+dy, fx+dx
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    score_vis[ny, nx] = color
    axes[2].imshow(score_vis, interpolation="nearest")
    axes[2].set_title("Step 3: Score frontiers\nby E[gain] + λ·Std", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    best_vis = vis.copy()
    by, bx = 14, 12
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny, nx = by+dy, bx+dx
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                best_vis[ny, nx] = [0, 1, 0]
    axes[3].imshow(best_vis, interpolation="nearest")
    axes[3].set_title("Step 4: Drive to best\nfrontier (PA3 planner)", fontsize=11, fontweight="bold")
    axes[3].plot(bx, by, "*", color="lime", markersize=20, markeredgecolor="black")
    axes[3].axis("off")

    plt.suptitle("How Frontier Scoring Works", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[3/7] Frontier scoring example saved: {save_path}")


def artifact_4_results_poster(eval_dir, history_path, save_path):
    """Single-image results poster for presentation."""
    with open(history_path) as f:
        h = json.load(f)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(h["epoch"], h["train_loss"], color="#2196F3", linewidth=2)
    val_data = [(e, v) for e, v in zip(h["epoch"], h["val_loss"]) if v is not None]
    if val_data:
        ax1.semilogy(*zip(*val_data), color="#FF9800", linewidth=2, marker="o", markersize=3)
    ax1.set_title("Loss Curve", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE (log)")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    metrics_data = {"IoU": 0.621, "Accuracy": 0.820, "Diversity": 0.223}
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    bars = ax2.bar(metrics_data.keys(), metrics_data.values(), color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, metrics_data.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", fontweight="bold", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title("Evaluation Metrics (K=8)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = fig.add_subplot(gs[0, 2])
    if os.path.exists(os.path.join(eval_dir, "metrics.csv")):
        ious = []
        with open(os.path.join(eval_dir, "metrics.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ious.append(float(row["mean_iou"]))
        ax3.hist(ious, bins=20, color="#4CAF50", edgecolor="white", alpha=0.8)
        ax3.axvline(np.mean(ious), color="red", linestyle="--", label=f"Mean: {np.mean(ious):.3f}")
        ax3.legend()
    ax3.set_title("IoU Distribution (n=100)", fontweight="bold")
    ax3.set_xlabel("IoU")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    summary = [
        ["Component", "Detail"],
        ["Model", "Conditional U-Net, 4.16M parameters"],
        ["Training", "DDPM, 29 epochs, batch=32, lr=1e-4, T=1000"],
        ["Data", "HouseExpo 35,126 maps, 2.66M augmented pairs"],
        ["Inference", "DDIM 50-step sampling, K=8 completions"],
        ["Final Loss", f"Train: {h['train_loss'][-1]:.5f}"],
        ["Evaluation", "IoU: 0.621, Accuracy: 82.0%, 100 test samples"],
        ["GPU", "Tesla T4 (g4dn.xlarge), ~10 hours training"],
    ]
    table = ax4.table(cellText=summary[1:], colLabels=summary[0],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2196F3")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        cell.set_edgecolor("#e0e0e0")

    fig.suptitle("Diffusion-Based Map Completion — Results Summary",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[4/7] Results poster saved: {save_path}")


def artifact_5_iou_vs_coverage(eval_dir, save_path):
    """Scatter plot: IoU vs partial map coverage with trend."""
    if not os.path.exists(os.path.join(eval_dir, "metrics.csv")):
        print("[5/7] Skipped (no metrics.csv)")
        return

    ious, coverages = [], []
    with open(os.path.join(eval_dir, "metrics.csv")) as f:
        for row in csv.DictReader(f):
            ious.append(float(row["mean_iou"]))
            coverages.append(float(row["coverage"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(coverages, ious, alpha=0.6, s=60, c=ious, cmap="RdYlGn",
               edgecolors="white", linewidth=0.5, vmin=0.3, vmax=0.9)

    z = np.polyfit(coverages, ious, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(coverages), max(coverages), 100)
    ax.plot(x_line, p(x_line), "--", color="red", linewidth=2,
            label=f"Trend: IoU = {z[0]:.2f} * coverage + {z[1]:.2f}")

    ax.set_xlabel("Partial Map Coverage (fraction visible)", fontsize=12)
    ax.set_ylabel("Prediction IoU", fontsize=12)
    ax.set_title("Does More Coverage = Better Predictions?", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[5/7] IoU vs coverage saved: {save_path}")


def artifact_6_loss_table(history_path, save_path):
    """Styled table of training milestones."""
    with open(history_path) as f:
        h = json.load(f)

    milestones = [1, 2, 5, 10, 15, 20, 25]
    rows = []
    for m in milestones:
        if m - 1 < len(h["epoch"]):
            idx = m - 1
            val_loss = h["val_loss"][idx]
            iou = h.get("iou", [None]*len(h["epoch"]))[idx]
            rows.append([
                f"Epoch {m}",
                f"{h['train_loss'][idx]:.5f}",
                f"{val_loss:.5f}" if val_loss else "-",
                f"{iou:.3f}" if iou else "-",
                f"{(1 - h['train_loss'][idx]/h['train_loss'][0])*100:.1f}%"
            ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    headers = ["Epoch", "Train Loss", "Val Loss", "Val IoU", "Improvement"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#E3F2FD")
        cell.set_edgecolor("#BBDEFB")

    plt.title("Training Milestones", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[6/7] Loss table saved: {save_path}")


def artifact_7_clean_strip(checkpoint_path, data_dir, device_str, save_path):
    """Single clean before/after strip for presentation header."""
    import torch
    from dataset import MapCompletionDataset
    from diffusion import DDPMScheduler
    from unet import ConditionalUNet

    device = torch.device(device_str)
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
    dataset = MapCompletionDataset(data_dir)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    labels = ["Ground Truth", "Partial (10%)", "Predicted", "Error", "Uncertainty"]

    sample = dataset[42]
    pm = sample["partial_map"].unsqueeze(0).to(device)
    km = sample["known_mask"].unsqueeze(0).to(device)
    fm = sample["full_map"].unsqueeze(0).to(device)

    with torch.no_grad():
        preds = []
        for _ in range(8):
            p = scheduler.sample_ddim(model, pm, km, num_steps=50)
            preds.append((p[0, 0].cpu().numpy() + 1) / 2)

    gt = (fm[0, 0].cpu().numpy() + 1) / 2
    partial = (pm[0, 0].cpu().numpy() + 1) / 2
    known = km[0, 0].cpu().numpy()
    mean_pred = np.mean(preds, axis=0).clip(0, 1)
    std_pred = np.std(preds, axis=0)

    axes[0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    vis = np.zeros((*partial.shape, 3))
    vis[partial > 0.6] = [1, 1, 1]
    vis[partial < 0.4] = [0, 0, 0]
    vis[(partial >= 0.4) & (partial <= 0.6)] = [0.5, 0.5, 0.7]
    axes[1].imshow(vis)
    axes[1].set_title(f"Partial ({known.mean():.0%} known)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1)
    iou = float(((gt > 0.5) & (mean_pred > 0.5)).sum()) / max(float(((gt > 0.5) | (mean_pred > 0.5)).sum()), 1)
    axes[2].set_title(f"Predicted (IoU: {iou:.2f})", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(np.abs(gt - mean_pred), cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("Prediction Error", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    im = axes[4].imshow(std_pred, cmap="hot", vmin=0, vmax=std_pred.max() + 0.001)
    axes[4].set_title("Model Uncertainty", fontsize=12, fontweight="bold")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[7/7] Clean strip saved: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, default="results/evaluation")
    parser.add_argument("--history", type=str, default="results/history.json")
    parser.add_argument("--out_dir", type=str, default="results/presentation")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    artifact_1_polished_loss_plot(args.history,
        os.path.join(args.out_dir, "01_loss_iou_plot.png"))
    artifact_2_pipeline_diagram(
        os.path.join(args.out_dir, "02_pipeline_diagram.png"))
    artifact_3_frontier_scoring_example(
        os.path.join(args.out_dir, "03_frontier_scoring.png"))
    artifact_4_results_poster(args.eval_dir, args.history,
        os.path.join(args.out_dir, "04_results_poster.png"))
    artifact_5_iou_vs_coverage(args.eval_dir,
        os.path.join(args.out_dir, "05_iou_vs_coverage.png"))
    artifact_6_loss_table(args.history,
        os.path.join(args.out_dir, "06_loss_table.png"))
    artifact_7_clean_strip(args.checkpoint, args.data_dir, args.device,
        os.path.join(args.out_dir, "07_clean_strip.png"))

    print(f"\nAll presentation artifacts saved to {args.out_dir}/")
