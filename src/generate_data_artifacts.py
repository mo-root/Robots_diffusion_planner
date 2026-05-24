"""Generate comprehensive visual artifacts for the data generation stage.

Produces:
1. Large sample grid (20+ partial/full pairs)
2. Augmentation showcase (one map, all 8 augmented variants)
3. Visibility ratio histogram
4. Map size/complexity distribution
5. Lidar raycasting visualization (showing rays)
6. Partial coverage progression (5%, 15%, 30%, 50%, 70%)
7. Dataset statistics summary card
"""

import json
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import (
    load_floor_plan, rasterize_floor_plan, find_free_position,
    simulate_lidar, make_partial_map, augment_pair
)


def artifact_1_large_sample_grid(data_dir, save_path, n=20):
    """Grid of 20 samples: partial map | ground truth | hidden regions."""
    files = sorted(Path(data_dir).glob("*.npz"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(files), size=min(n, len(files)), replace=False)

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))

    for idx, file_idx in enumerate(sorted(indices)):
        row = idx // cols
        col_base = (idx % cols) * 3

        data = np.load(files[file_idx])["data"].astype(np.float32)
        partial, known_mask, full_map = data[:,:,0], data[:,:,1], data[:,:,2]

        axes[row, col_base].imshow(full_map, cmap="gray_r", vmin=0, vmax=1)
        axes[row, col_base].set_title("GT", fontsize=8)
        axes[row, col_base].axis("off")

        vis = np.zeros((*partial.shape, 3))
        vis[partial > 0.7] = [1, 1, 1]
        vis[partial < 0.3] = [0, 0, 0]
        vis[(partial > 0.3) & (partial < 0.7)] = [0.5, 0.5, 0.7]
        axes[row, col_base + 1].imshow(vis)
        axes[row, col_base + 1].set_title(f"Partial ({known_mask.mean():.0%})", fontsize=8)
        axes[row, col_base + 1].axis("off")

        diff = np.zeros((*partial.shape, 3))
        diff[full_map > 0.5] = [0.9, 0.9, 0.9]
        diff[(full_map > 0.5) & (known_mask < 0.5)] = [0.3, 0.8, 0.3]
        diff[(full_map < 0.5) & (known_mask < 0.5)] = [0.8, 0.3, 0.3]
        diff[(full_map < 0.5) & (known_mask > 0.5)] = [0, 0, 0]
        axes[row, col_base + 2].imshow(diff)
        axes[row, col_base + 2].set_title("Hidden", fontsize=8)
        axes[row, col_base + 2].axis("off")

    plt.suptitle(f"Training Data Samples ({n} random pairs)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[1/7] Sample grid saved: {save_path}")


def artifact_2_augmentation_showcase(json_dir, save_path):
    """Show one floor plan with all 8 augmentation variants."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    plan = load_floor_plan(str(json_files[100]))
    grid = rasterize_floor_plan(plan, 256)
    rng = np.random.default_rng(42)
    pos = find_free_position(grid, rng)
    visible = simulate_lidar(grid, pos, num_rays=360, max_range_px=70)
    sample = make_partial_map(grid, visible)
    augmented = augment_pair(sample)

    labels = ["Original", "Rot 90", "Rot 180", "Rot 270",
              "Flip H", "Flip H + Rot 90", "Flip H + Rot 180", "Flip H + Rot 270"]

    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for i, (aug, label) in enumerate(zip(augmented, labels)):
        partial = aug[:,:,0]
        full = aug[:,:,2]

        vis = np.zeros((*partial.shape, 3))
        vis[partial > 0.7] = [1, 1, 1]
        vis[partial < 0.3] = [0, 0, 0]
        vis[(partial > 0.3) & (partial < 0.7)] = [0.5, 0.5, 0.7]

        axes[0, i].imshow(vis)
        axes[0, i].set_title(label, fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(full, cmap="gray_r", vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Partial", fontsize=11)
    axes[1, 0].set_ylabel("Full", fontsize=11)
    plt.suptitle("Data Augmentation: 8 variants from one floor plan + one robot position",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[2/7] Augmentation showcase saved: {save_path}")


def artifact_3_visibility_histogram(data_dir, save_path, max_files=2000):
    """Histogram of how much of the map is visible (partial coverage distribution)."""
    files = sorted(Path(data_dir).glob("*.npz"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(files), size=min(max_files, len(files)), replace=False)

    ratios = []
    for idx in indices:
        data = np.load(files[idx])["data"].astype(np.float32)
        known_mask = data[:,:,1]
        full_map = data[:,:,2]
        free_cells = (full_map > 0.5).sum()
        if free_cells > 0:
            ratios.append(known_mask.sum() / full_map.size)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ratios, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(ratios), color="red", linestyle="--", label=f"Mean: {np.mean(ratios):.1%}")
    ax.axvline(np.median(ratios), color="orange", linestyle="--", label=f"Median: {np.median(ratios):.1%}")
    ax.set_xlabel("Fraction of map visible", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of partial map coverage across training samples", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[3/7] Visibility histogram saved: {save_path}")


def artifact_4_map_complexity(json_dir, save_path, max_maps=5000):
    """Distribution of room counts and map sizes."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(json_files), size=min(max_maps, len(json_files)), replace=False)

    room_counts = []
    areas = []
    vert_counts = []

    for idx in indices:
        plan = load_floor_plan(str(json_files[idx]))
        room_counts.append(plan.get("room_num", 0))
        bbox = plan.get("bbox", {})
        w = bbox.get("max", [0,0])[0] - bbox.get("min", [0,0])[0]
        h = bbox.get("max", [0,0])[1] - bbox.get("min", [0,0])[1]
        areas.append(w * h)
        vert_counts.append(len(plan.get("verts", [])))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(room_counts, bins=range(1, max(room_counts)+2), color="#DD8452",
                 edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Number of rooms")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Room count distribution\n(mean: {np.mean(room_counts):.1f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(areas, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Floor area (m^2)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Floor plan area\n(mean: {np.mean(areas):.0f} m^2)")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(vert_counts, bins=50, color="#C44E52", edgecolor="white", alpha=0.8)
    axes[2].set_xlabel("Number of vertices")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Wall complexity\n(mean: {np.mean(vert_counts):.0f} vertices)")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("HouseExpo Dataset Statistics (35,126 floor plans)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[4/7] Map complexity distribution saved: {save_path}")


def artifact_5_lidar_raycasting_viz(json_dir, save_path):
    """Visualize the lidar raycasting process with visible rays."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    plan = load_floor_plan(str(json_files[200]))
    grid = rasterize_floor_plan(plan, 256)
    rng = np.random.default_rng(42)
    pos = find_free_position(grid, rng)
    ry, rx = pos

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    axes[0].plot(rx, ry, "ro", markersize=8)
    axes[0].set_title("Floor plan + robot position")
    axes[0].axis("off")

    ray_img = np.stack([grid, grid, grid], axis=-1)
    num_rays = 360
    max_range = 70
    for i in range(0, num_rays, 3):
        angle = 2.0 * math.pi * i / num_rays
        dx, dy = math.cos(angle), math.sin(angle)
        for step in range(1, max_range + 1):
            xi = int(round(rx + dx * step))
            yi = int(round(ry + dy * step))
            if xi < 0 or xi >= 256 or yi < 0 or yi >= 256:
                break
            ray_img[yi, xi] = [0.2, 0.6, 1.0]
            if grid[yi, xi] < 0.5:
                ray_img[yi, xi] = [1.0, 0.3, 0.3]
                break

    axes[1].imshow(ray_img)
    axes[1].plot(rx, ry, "ro", markersize=8)
    axes[1].set_title("Lidar rays (blue=free, red=hit wall)")
    axes[1].axis("off")

    visible = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_range)
    vis_overlay = np.zeros((256, 256, 3))
    vis_overlay[visible > 0] = [0.2, 0.8, 0.2]
    vis_overlay[grid < 0.5] = [0, 0, 0]
    vis_overlay[(visible > 0) & (grid < 0.5)] = [0.8, 0.2, 0.2]
    axes[2].imshow(vis_overlay)
    axes[2].plot(rx, ry, "ro", markersize=8)
    axes[2].set_title("Visibility mask (green=seen free)")
    axes[2].axis("off")

    partial = make_partial_map(grid, visible)[:,:,0]
    vis = np.zeros((*partial.shape, 3))
    vis[partial > 0.7] = [1, 1, 1]
    vis[partial < 0.3] = [0, 0, 0]
    vis[(partial > 0.3) & (partial < 0.7)] = [0.5, 0.5, 0.7]
    axes[3].imshow(vis)
    axes[3].set_title("Resulting partial map")
    axes[3].axis("off")

    plt.suptitle("Lidar Raycasting Simulation Pipeline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[5/7] Lidar raycasting viz saved: {save_path}")


def artifact_6_coverage_progression(json_dir, save_path):
    """Show same map at different coverage levels (multiple robot positions)."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    plan = load_floor_plan(str(json_files[300]))
    grid = rasterize_floor_plan(plan, 256)
    rng = np.random.default_rng(42)

    ranges = [30, 50, 70, 90, 120]
    labels = []

    fig, axes = plt.subplots(1, len(ranges) + 1, figsize=(4 * (len(ranges) + 1), 4))

    axes[0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].axis("off")

    for i, max_r in enumerate(ranges):
        pos = find_free_position(grid, rng)
        visible = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_r)
        partial = make_partial_map(grid, visible)[:,:,0]
        known = make_partial_map(grid, visible)[:,:,1]
        coverage = known.mean()

        vis = np.zeros((*partial.shape, 3))
        vis[partial > 0.7] = [1, 1, 1]
        vis[partial < 0.3] = [0, 0, 0]
        vis[(partial > 0.3) & (partial < 0.7)] = [0.5, 0.5, 0.7]

        axes[i+1].imshow(vis)
        axes[i+1].plot(pos[1], pos[0], "ro", markersize=6)
        axes[i+1].set_title(f"Range={max_r}px\n({coverage:.0%} known)")
        axes[i+1].axis("off")

    plt.suptitle("Partial Map Coverage vs Lidar Range", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[6/7] Coverage progression saved: {save_path}")


def artifact_7_stats_card(data_dir, json_dir, save_path):
    """Summary statistics card."""
    json_files = list(Path(json_dir).glob("*.json"))
    npz_files = list(Path(data_dir).glob("*.npz"))

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(npz_files), size=min(500, len(npz_files)), replace=False)
    coverages = []
    for idx in sample_indices:
        data = np.load(npz_files[idx])["data"].astype(np.float32)
        coverages.append(data[:,:,1].mean())

    stats = {
        "Total floor plans": f"{len(json_files):,}",
        "Training samples": f"{len(npz_files):,}",
        "Resolution": "256 x 256",
        "Channels": "3 (partial, mask, full)",
        "Augmentation": "8x (4 rotations x 2 flips)",
        "Samples per map": "10",
        "Lidar range": "40-100 px (randomized)",
        "Lidar rays": "360",
        "Mean coverage": f"{np.mean(coverages):.1%}",
        "Coverage std": f"{np.std(coverages):.1%}",
        "Storage format": ".npz (float16)",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    table_data = [[k, v] for k, v in stats.items()]
    table = ax.table(cellText=table_data, colLabels=["Property", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    plt.title("Dataset Statistics", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[7/7] Stats card saved: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/data_artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    artifact_1_large_sample_grid(args.data_dir,
        os.path.join(args.out_dir, "01_sample_grid_20.png"))
    artifact_2_augmentation_showcase(args.json_dir,
        os.path.join(args.out_dir, "02_augmentation_showcase.png"))
    artifact_3_visibility_histogram(args.data_dir,
        os.path.join(args.out_dir, "03_visibility_histogram.png"))
    artifact_4_map_complexity(args.json_dir,
        os.path.join(args.out_dir, "04_map_complexity.png"))
    artifact_5_lidar_raycasting_viz(args.json_dir,
        os.path.join(args.out_dir, "05_lidar_raycasting.png"))
    artifact_6_coverage_progression(args.json_dir,
        os.path.join(args.out_dir, "06_coverage_progression.png"))
    artifact_7_stats_card(args.data_dir, args.json_dir,
        os.path.join(args.out_dir, "07_dataset_stats.png"))

    print(f"\nAll artifacts saved to {args.out_dir}/")
