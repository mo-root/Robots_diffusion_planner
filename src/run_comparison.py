"""Offline comparison: diffusion frontier scoring vs heuristic baseline.

Runs both scorers on the same set of partial maps and compares which
frontiers they select and the expected information gain.

This doesn't need ROS -- it runs the scoring logic directly on saved maps.
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


def get_multi_scan(grid, n_pos, rng, max_range=70):
    h, w = grid.shape
    combined = np.zeros((h, w), dtype=np.uint8)
    positions = []
    for _ in range(n_pos):
        pos = find_free_position(grid, rng)
        if pos:
            vis = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_range)
            combined = np.maximum(combined, vis)
            positions.append(pos)
    partial = np.full_like(grid, 0.5)
    partial[combined > 0] = grid[combined > 0]
    known = (combined > 0).astype(np.float32)
    return partial, known, positions[-1] if positions else (128, 128)


def detect_frontier_clusters(known_mask, min_size=5):
    h, w = known_mask.shape
    frontiers = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if known_mask[y, x] < 0.5:
                continue
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx] < 0.5:
                    frontiers.append((y, x))
                    break

    if not frontiers:
        return []

    frontier_set = set(frontiers)
    visited = set()
    clusters = []
    for f in frontiers:
        if f in visited:
            continue
        cluster = []
        stack = [f]
        while stack:
            cell = stack.pop()
            if cell in visited:
                continue
            visited.add(cell)
            cluster.append(cell)
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                n = (cell[0]+dy, cell[1]+dx)
                if n in frontier_set and n not in visited:
                    stack.append(n)
        if len(cluster) >= min_size:
            cy = int(np.mean([c[0] for c in cluster]))
            cx = int(np.mean([c[1] for c in cluster]))
            clusters.append((cy, cx, len(cluster)))
    return clusters


def baseline_score(clusters, known_mask, robot_pos, info_radius=15, beta=0.5):
    h, w = known_mask.shape
    ry, rx = robot_pos
    scored = []
    for cy, cx, size in clusters:
        r = info_radius
        y1, y2 = max(0, cy-r), min(h, cy+r+1)
        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        unknown = float((known_mask[y1:y2, x1:x2] < 0.5).sum())
        dist = math.sqrt((cy-ry)**2 + (cx-rx)**2)
        score = unknown - beta * dist
        scored.append((score, cy, cx))
    scored.sort(reverse=True)
    return scored


@torch.no_grad()
def diffusion_score(model, scheduler, clusters, partial, known_mask, robot_pos,
                    device, K=8, info_radius=15, lambda_var=1.0, beta=0.5):
    pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    completions = []
    for _ in range(K):
        pred = scheduler.sample_ddim(model, pm, km, num_steps=50)
        completions.append((pred[0, 0].cpu().numpy() + 1) / 2)

    h, w = known_mask.shape
    ry, rx = robot_pos
    scored = []
    for cy, cx, size in clusters:
        gains = []
        for comp in completions:
            r = info_radius
            y1, y2 = max(0, cy-r), min(h, cy+r+1)
            x1, x2 = max(0, cx-r), min(w, cx+r+1)
            gain = float(((comp[y1:y2, x1:x2] > 0.5) & (known_mask[y1:y2, x1:x2] < 0.5)).sum())
            gains.append(gain)
        expected = np.mean(gains)
        variance = np.std(gains)
        dist = math.sqrt((cy-ry)**2 + (cx-rx)**2)
        score = expected + lambda_var * variance - beta * dist
        scored.append((score, cy, cx))

    scored.sort(reverse=True)
    return scored, completions


def actual_info_gain(grid, known_mask, frontier_pos, radius=15):
    """Ground truth: how much free space is actually near this frontier."""
    fy, fx = frontier_pos
    h, w = grid.shape
    y1, y2 = max(0, fy-radius), min(h, fy+radius+1)
    x1, x2 = max(0, fx-radius), min(w, fx+radius+1)
    return float(((grid[y1:y2, x1:x2] > 0.5) & (known_mask[y1:y2, x1:x2] < 0.5)).sum())


def partial_rgb(p):
    vis = np.zeros((*p.shape, 3))
    vis[p > 0.6] = [1, 1, 1]
    vis[p < 0.4] = [0, 0, 0]
    vis[(p >= 0.4) & (p <= 0.6)] = [0.5, 0.5, 0.7]
    return vis


def run_comparison(model, scheduler, json_dir, device, out_dir, n_maps=10, n_scans=3):
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(json_files), size=n_maps, replace=False)

    baseline_gains = []
    diffusion_gains = []

    fig, axes = plt.subplots(n_maps, 4, figsize=(20, 4 * n_maps))

    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        partial, known, robot_pos = get_multi_scan(grid, n_scans, np.random.default_rng(row * 7))
        clusters = detect_frontier_clusters(known)

        if not clusters:
            continue

        b_scored = baseline_score(clusters, known, robot_pos)
        d_scored, completions = diffusion_score(
            model, scheduler, clusters, partial, known, robot_pos, device)

        b_best = (b_scored[0][1], b_scored[0][2])
        d_best = (d_scored[0][1], d_scored[0][2])

        b_actual = actual_info_gain(grid, known, b_best)
        d_actual = actual_info_gain(grid, known, d_best)
        baseline_gains.append(b_actual)
        diffusion_gains.append(d_actual)

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("Ground Truth" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_rgb(partial))
        axes[row, 1].plot(robot_pos[1], robot_pos[0], "ro", markersize=6)
        axes[row, 1].set_title(f"Partial ({known.mean():.0%})" if row == 0 else f"({known.mean():.0%})")
        axes[row, 1].axis("off")

        goal_vis = partial_rgb(partial).copy()
        goal_vis[max(0,b_best[0]-3):b_best[0]+4, max(0,b_best[1]-3):b_best[1]+4] = [1, 0.5, 0]
        goal_vis[max(0,d_best[0]-3):d_best[0]+4, max(0,d_best[1]-3):d_best[1]+4] = [0, 1, 0]
        axes[row, 2].imshow(goal_vis)
        axes[row, 2].plot(b_best[1], b_best[0], "s", color="orange", markersize=10, label="Baseline")
        axes[row, 2].plot(d_best[1], d_best[0], "*", color="lime", markersize=14, label="Diffusion")
        if row == 0:
            axes[row, 2].legend(fontsize=9)
        axes[row, 2].set_title("Frontier choices" if row == 0 else "")
        axes[row, 2].axis("off")

        mean_comp = np.mean(completions, axis=0).clip(0, 1)
        axes[row, 3].imshow(mean_comp, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 3].plot(d_best[1], d_best[0], "*", color="lime", markersize=12)
        win = "Diff" if d_actual > b_actual else "Base" if b_actual > d_actual else "Tie"
        color = "green" if win == "Diff" else "red" if win == "Base" else "gray"
        axes[row, 3].set_title(
            f"B:{b_actual:.0f} vs D:{d_actual:.0f} ({win})" if row == 0
            else f"B:{b_actual:.0f} vs D:{d_actual:.0f}",
            color=color, fontweight="bold")
        axes[row, 3].axis("off")

    plt.suptitle("Baseline (orange) vs Diffusion (green) Frontier Selection\n"
                 "Numbers show actual info gain at chosen frontier",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(baseline_gains))
    axes2[0].bar([i - 0.15 for i in x], baseline_gains, width=0.3,
                  color="#FF9800", label="Baseline", alpha=0.8)
    axes2[0].bar([i + 0.15 for i in x], diffusion_gains, width=0.3,
                  color="#4CAF50", label="Diffusion", alpha=0.8)
    axes2[0].set_xlabel("Map #")
    axes2[0].set_ylabel("Actual Info Gain at Chosen Frontier")
    axes2[0].set_title("Per-Map Comparison", fontweight="bold")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3, axis="y")

    wins_diff = sum(d > b for d, b in zip(diffusion_gains, baseline_gains))
    wins_base = sum(b > d for d, b in zip(diffusion_gains, baseline_gains))
    ties = sum(d == b for d, b in zip(diffusion_gains, baseline_gains))

    labels = ["Diffusion\nWins", "Baseline\nWins", "Ties"]
    counts = [wins_diff, wins_base, ties]
    colors = ["#4CAF50", "#FF9800", "#9E9E9E"]
    axes2[1].bar(labels, counts, color=colors, edgecolor="white", linewidth=2)
    for i, c in enumerate(counts):
        axes2[1].text(i, c + 0.2, str(c), ha="center", fontweight="bold", fontsize=14)
    axes2[1].set_title("Win Rate", fontweight="bold")
    axes2[1].set_ylabel("Count")

    avg_b = np.mean(baseline_gains)
    avg_d = np.mean(diffusion_gains)
    improvement = (avg_d - avg_b) / max(avg_b, 1) * 100

    plt.suptitle(f"Diffusion vs Baseline: Avg gain {avg_d:.0f} vs {avg_b:.0f} "
                 f"({improvement:+.0f}%)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nResults:")
    print(f"  Baseline avg info gain: {avg_b:.1f}")
    print(f"  Diffusion avg info gain: {avg_d:.1f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Diffusion wins: {wins_diff}/{len(baseline_gains)}")
    print(f"  Baseline wins: {wins_base}/{len(baseline_gains)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/comparison")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--n_maps", type=int, default=10)
    parser.add_argument("--n_scans", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)

    print(f"Running comparison on {args.n_maps} maps ({args.n_scans} scans each)...")
    run_comparison(model, scheduler, args.json_dir, device, args.out_dir,
                   n_maps=args.n_maps, n_scans=args.n_scans)
    print(f"\nResults saved to {args.out_dir}/")
