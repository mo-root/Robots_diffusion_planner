"""Generate rich visual artifacts showing how predictions improve with more coverage.

Produces:
1. Progressive exploration GIF (robot explores, predictions improve)
2. Side-by-side coverage strip (1 scan vs 3 vs 5 vs 8 -- same map)
3. Coverage sweep GIF (smoothly increasing coverage, watch prediction sharpen)
4. Multi-map coverage comparison (6 maps, each at 3 coverage levels)
5. Prediction confidence heatmap at different coverages
6. "Robot exploration story" -- narrative strip showing the full exploration loop
"""

import os
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from diffusion import DDPMScheduler
from unet import ConditionalUNet

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL required")


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


def get_cumulative_scans(grid, n_total, rng, max_range=70):
    """Return list of (partial, known_mask) with progressively more scans."""
    h, w = grid.shape
    combined = np.zeros((h, w), dtype=np.uint8)
    results = []

    for i in range(n_total):
        pos = find_free_position(grid, rng)
        if pos is None:
            continue
        visible = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_range)
        combined = np.maximum(combined, visible)

        partial = np.full_like(grid, 0.5)
        partial[combined > 0] = grid[combined > 0]
        known = (combined > 0).astype(np.float32)
        results.append((partial.copy(), known.copy(), pos))

    return results


@torch.no_grad()
def predict(model, scheduler, partial, known_mask, device):
    pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    pred = scheduler.sample_ddim(model, pm, km, num_steps=50)
    return (pred[0, 0].cpu().numpy() + 1) / 2


@torch.no_grad()
def predict_k(model, scheduler, partial, known_mask, device, K=8):
    pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    preds = []
    for _ in range(K):
        pred = scheduler.sample_ddim(model, pm, km, num_steps=50)
        preds.append((pred[0, 0].cpu().numpy() + 1) / 2)
    return preds


def artifact_1_exploration_gif(model, scheduler, grid, device, save_path, n_scans=10):
    """GIF: robot explores step by step, predictions improve."""
    rng = np.random.default_rng(42)
    scans = get_cumulative_scans(grid, n_scans, rng)

    frames = []
    for i, (partial, known, robot_pos) in enumerate(scans):
        predicted = predict(model, scheduler, partial, known, device).clip(0, 1)
        score = iou(grid, predicted)
        coverage = known.mean()

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        axes[0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[0].plot(robot_pos[1], robot_pos[0], "ro", markersize=8)
        axes[0].set_title("Ground Truth + Robot", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(partial_rgb(partial))
        axes[1].plot(robot_pos[1], robot_pos[0], "ro", markersize=8)
        axes[1].set_title(f"Explored: {coverage:.0%} ({i+1} scans)", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[2].set_title(f"Predicted Map (IoU: {score:.2f})", fontsize=12)
        axes[2].axis("off")

        error = np.abs(grid - predicted)
        axes[3].imshow(error, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Prediction Error", fontsize=12)
        axes[3].axis("off")

        plt.suptitle(f"Exploration Step {i+1}/{n_scans}: Watch predictions improve as robot explores",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close()

    for _ in range(10):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=800, loop=0)
    print(f"[1/6] Exploration GIF saved: {save_path} ({len(frames)} frames)")


def artifact_2_coverage_strip(model, scheduler, grid, device, save_path):
    """Single image: same map at 1, 3, 5, 8 scans side by side."""
    rng = np.random.default_rng(55)
    scans = get_cumulative_scans(grid, 8, rng)

    pick_indices = [0, 2, 4, 7]
    fig, axes = plt.subplots(2, len(pick_indices) + 1, figsize=(4 * (len(pick_indices) + 1), 8))

    axes[0, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Ground\nTruth", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, idx in enumerate(pick_indices):
        partial, known, _ = scans[idx]
        predicted = predict(model, scheduler, partial, known, device).clip(0, 1)
        score = iou(grid, predicted)
        coverage = known.mean()

        axes[0, col+1].imshow(partial_rgb(partial))
        axes[0, col+1].set_title(f"{idx+1} scan{'s' if idx > 0 else ''}\n({coverage:.0%} known)",
                                  fontsize=11, fontweight="bold")
        axes[0, col+1].axis("off")

        axes[1, col+1].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[1, col+1].set_title(f"IoU: {score:.2f}", fontsize=12,
                                  color="green" if score > 0.7 else "orange" if score > 0.5 else "red")
        axes[1, col+1].axis("off")

    axes[0, 0].set_ylabel("Input", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Prediction", fontsize=13, fontweight="bold")

    plt.suptitle("Same Floor Plan at Different Exploration Stages",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[2/6] Coverage strip saved: {save_path}")


def artifact_3_sweep_gif(model, scheduler, grid, device, save_path):
    """GIF: smoothly increasing coverage from 1 to 12 scans."""
    rng = np.random.default_rng(77)
    scans = get_cumulative_scans(grid, 12, rng)

    frames = []
    ious_so_far = []

    for i, (partial, known, _) in enumerate(scans):
        predicted = predict(model, scheduler, partial, known, device).clip(0, 1)
        score = iou(grid, predicted)
        ious_so_far.append(score)
        coverage = known.mean()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(partial_rgb(partial))
        axes[0].set_title(f"Partial Map ({coverage:.0%} known)", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"Prediction (IoU: {score:.2f})", fontsize=13)
        axes[1].axis("off")

        axes[2].bar(range(len(ious_so_far)), ious_so_far,
                    color=["#4CAF50" if s > 0.7 else "#FF9800" if s > 0.5 else "#f44336" for s in ious_so_far])
        axes[2].set_xlim(-0.5, 11.5)
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Scan #")
        axes[2].set_ylabel("IoU")
        axes[2].set_title("IoU Over Exploration", fontsize=13)
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"Scan {i+1}/12: More exploration = better predictions",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        frames.append(fig_to_frame(fig))
        plt.close()

    for _ in range(8):
        frames.append(frames[-1])

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=700, loop=0)
    print(f"[3/6] Sweep GIF saved: {save_path} ({len(frames)} frames)")


def artifact_4_multi_map_coverage(model, scheduler, json_dir, device, save_path, n_maps=6):
    """6 different maps, each at 1 scan, 4 scans, 8 scans."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(33)
    indices = rng.choice(len(json_files), size=n_maps, replace=False)
    coverages = [1, 4, 8]

    fig, axes = plt.subplots(n_maps, len(coverages) * 2 + 1, figsize=(3 * (len(coverages) * 2 + 1), 3.5 * n_maps))

    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("GT" if row == 0 else "")
        axes[row, 0].axis("off")

        for ci, n_scans in enumerate(coverages):
            scan_rng = np.random.default_rng(row * 100 + ci)
            scans = get_cumulative_scans(grid, n_scans, scan_rng)
            partial, known, _ = scans[-1]
            predicted = predict(model, scheduler, partial, known, device).clip(0, 1)
            score = iou(grid, predicted)

            col_input = 1 + ci * 2
            col_pred = 2 + ci * 2

            axes[row, col_input].imshow(partial_rgb(partial))
            axes[row, col_input].set_title(f"{n_scans}x ({known.mean():.0%})" if row == 0 else f"({known.mean():.0%})")
            axes[row, col_input].axis("off")

            axes[row, col_pred].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
            color = "green" if score > 0.7 else "orange" if score > 0.5 else "red"
            axes[row, col_pred].set_title(f"{score:.2f}" if row > 0 else f"IoU: {score:.2f}",
                                           fontsize=11, color=color, fontweight="bold")
            axes[row, col_pred].axis("off")

    plt.suptitle("6 Floor Plans x 3 Coverage Levels: Predictions improve consistently",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[4/6] Multi-map coverage saved: {save_path}")


def artifact_5_confidence_heatmap(model, scheduler, grid, device, save_path, K=8):
    """Show uncertainty decreasing as coverage increases."""
    rng = np.random.default_rng(88)
    scans = get_cumulative_scans(grid, 8, rng)
    pick = [0, 2, 4, 7]

    fig, axes = plt.subplots(3, len(pick), figsize=(5 * len(pick), 13))

    for col, idx in enumerate(pick):
        partial, known, _ = scans[idx]
        preds = predict_k(model, scheduler, partial, known, device, K=K)
        mean_pred = np.mean(preds, axis=0).clip(0, 1)
        std_pred = np.std(preds, axis=0)

        axes[0, col].imshow(partial_rgb(partial))
        axes[0, col].set_title(f"{idx+1} scan{'s' if idx > 0 else ''} ({known.mean():.0%})",
                                fontsize=12, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1)
        score = iou(grid, mean_pred)
        axes[1, col].set_title(f"Mean Pred (IoU: {score:.2f})", fontsize=11)
        axes[1, col].axis("off")

        im = axes[2, col].imshow(std_pred, cmap="hot", vmin=0, vmax=0.4)
        axes[2, col].set_title(f"Uncertainty (avg: {std_pred.mean():.3f})", fontsize=11)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Input", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Prediction", fontsize=13, fontweight="bold")
    axes[2, 0].set_ylabel("Uncertainty", fontsize=13, fontweight="bold")

    plt.suptitle("Model Confidence Increases with More Coverage\n"
                 "Uncertainty (bright) shrinks as robot explores more",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[5/6] Confidence heatmap saved: {save_path}")


def artifact_6_exploration_story(model, scheduler, grid, device, save_path):
    """Narrative strip: the full exploration loop as a robot would experience it."""
    rng = np.random.default_rng(42)
    scans = get_cumulative_scans(grid, 8, rng)
    steps = [0, 1, 3, 5, 7]

    fig, axes = plt.subplots(3, len(steps), figsize=(5 * len(steps), 12))

    titles_row0 = ["Robot starts\nexploring", "Second scan\nnew area found",
                   "4th scan\nhallway mapped", "6th scan\nmost rooms found",
                   "8th scan\nalmost complete"]

    for col, (step, title) in enumerate(zip(steps, titles_row0)):
        partial, known, pos = scans[step]
        preds = predict_k(model, scheduler, partial, known, device, K=4)
        mean_pred = np.mean(preds, axis=0).clip(0, 1)
        std_pred = np.std(preds, axis=0)
        score = iou(grid, mean_pred)

        axes[0, col].imshow(partial_rgb(partial))
        axes[0, col].plot(pos[1], pos[0], "ro", markersize=6)
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1)
        axes[1, col].set_title(f"IoU: {score:.2f}", fontsize=12,
                                color="green" if score > 0.7 else "orange")
        axes[1, col].axis("off")

        overlay = np.zeros((*grid.shape, 3))
        overlay[:] = [0.5, 0.5, 0.5]
        gt_bin = grid > 0.5
        pred_bin = mean_pred > 0.5
        overlay[gt_bin & pred_bin] = [0.9, 0.9, 0.9]
        overlay[gt_bin & ~pred_bin & (known < 0.5)] = [0.3, 0.3, 0.9]
        overlay[~gt_bin & pred_bin & (known < 0.5)] = [0.9, 0.3, 0.3]
        overlay[~gt_bin & ~pred_bin] = [0.15, 0.15, 0.15]
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("white=correct, blue=missed, red=false", fontsize=9)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("What robot\nsees", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("What model\npredicts", fontsize=12, fontweight="bold")
    axes[2, 0].set_ylabel("Accuracy\nmap", fontsize=12, fontweight="bold")

    plt.suptitle("The Exploration Story: From First Scan to Nearly Complete Map",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[6/6] Exploration story saved: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/coverage_artifacts")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--map_index", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    plan = load_floor_plan(str(json_files[args.map_index]))
    grid = rasterize_floor_plan(plan, 256)
    print(f"Using map {args.map_index} ({plan.get('room_num', '?')} rooms)\n")

    artifact_1_exploration_gif(model, scheduler, grid, device,
                                os.path.join(args.out_dir, "01_exploration.gif"))
    artifact_2_coverage_strip(model, scheduler, grid, device,
                               os.path.join(args.out_dir, "02_coverage_strip.png"))
    artifact_3_sweep_gif(model, scheduler, grid, device,
                          os.path.join(args.out_dir, "03_coverage_sweep.gif"))
    artifact_4_multi_map_coverage(model, scheduler, args.json_dir, device,
                                   os.path.join(args.out_dir, "04_multi_map.png"))
    artifact_5_confidence_heatmap(model, scheduler, grid, device,
                                   os.path.join(args.out_dir, "05_confidence.png"))
    artifact_6_exploration_story(model, scheduler, grid, device,
                                  os.path.join(args.out_dir, "06_exploration_story.png"))

    print(f"\nAll coverage artifacts saved to {args.out_dir}/")
