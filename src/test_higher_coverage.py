"""Test model predictions at different input coverage levels.

Generates partial maps with 1, 2, 3, 5 robot positions (simulating
a robot that has been exploring for longer) and compares prediction quality.
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


def make_multi_scan_partial(grid, n_positions, rng, max_range=70):
    """Simulate n_positions robot scans and union their visibility."""
    h, w = grid.shape
    combined_visible = np.zeros((h, w), dtype=np.uint8)

    for _ in range(n_positions):
        pos = find_free_position(grid, rng)
        if pos is None:
            continue
        visible = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_range)
        combined_visible = np.maximum(combined_visible, visible)

    partial = np.full_like(grid, 0.5)
    partial[combined_visible > 0] = grid[combined_visible > 0]
    known_mask = (combined_visible > 0).astype(np.float32)

    return partial, known_mask


def partial_rgb(p):
    vis = np.zeros((*p.shape, 3))
    vis[p > 0.6] = [1, 1, 1]
    vis[p < 0.4] = [0, 0, 0]
    vis[(p >= 0.4) & (p <= 0.6)] = [0.5, 0.5, 0.7]
    return vis


def iou(gt, pred):
    g, p = gt > 0.5, pred > 0.5
    return float((g & p).sum()) / max(float((g | p).sum()), 1)


@torch.no_grad()
def test_coverage_levels(model, scheduler, json_dir, device, save_path,
                         n_maps=6, positions_list=[1, 2, 3, 5, 8]):
    """Test predictions at different coverage levels."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(json_files), size=n_maps, replace=False)

    fig, axes = plt.subplots(n_maps, len(positions_list) + 1,
                              figsize=(4 * (len(positions_list) + 1), 4 * n_maps))

    all_ious = {n: [] for n in positions_list}

    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("Ground Truth" if row == 0 else "")
        axes[row, 0].axis("off")

        for col, n_pos in enumerate(positions_list):
            partial, known_mask = make_multi_scan_partial(grid, n_pos, rng)
            coverage = known_mask.mean()

            pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pred = scheduler.sample_ddim(model, pm, km, num_steps=50)
            pred_np = (pred[0, 0].cpu().numpy() + 1) / 2
            pred_np = pred_np.clip(0, 1)

            score = iou(grid, pred_np)
            all_ious[n_pos].append(score)

            combined = np.zeros((*grid.shape, 3))
            combined[:] = pred_np[:, :, None] * [1, 1, 1]
            combined[known_mask > 0.5] = partial_rgb(partial)[known_mask > 0.5]

            axes[row, col + 1].imshow(pred_np, cmap="gray_r", vmin=0, vmax=1)
            title = f"{n_pos} scan{'s' if n_pos > 1 else ''} ({coverage:.0%})" if row == 0 else f"({coverage:.0%})"
            axes[row, col + 1].set_title(f"{title}\nIoU: {score:.2f}", fontsize=10)
            axes[row, col + 1].axis("off")

    plt.suptitle("Effect of Input Coverage on Prediction Quality\n"
                 "More robot scans = more visible map = better predictions",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Coverage comparison saved: {save_path}")

    fig2, ax = plt.subplots(figsize=(10, 6))
    means = [np.mean(all_ious[n]) for n in positions_list]
    stds = [np.std(all_ious[n]) for n in positions_list]
    labels = [f"{n} scan{'s' if n > 1 else ''}" for n in positions_list]

    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=["#ef5350", "#ff9800", "#ffeb3b", "#66bb6a", "#26a69a"],
                  edgecolor="white", linewidth=2)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", fontweight="bold", fontsize=13)

    ax.set_ylabel("Mean IoU", fontsize=13)
    ax.set_xlabel("Number of Robot Scans (input coverage)", fontsize=13)
    ax.set_title("Prediction Quality vs Input Coverage", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    bar_path = save_path.replace(".png", "_bar.png")
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Coverage bar chart saved: {bar_path}")

    return all_ious


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/coverage_analysis")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)

    print("Testing coverage levels...")
    ious = test_coverage_levels(
        model, scheduler, args.json_dir, device,
        os.path.join(args.out_dir, "coverage_comparison.png"),
        n_maps=6, positions_list=[1, 2, 3, 5, 8]
    )

    print("\nResults:")
    for n, scores in ious.items():
        print(f"  {n} scan{'s' if n > 1 else ''}: IoU = {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
