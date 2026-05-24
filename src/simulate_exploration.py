"""Simulate full diffusion-guided exploration and produce a demo video.

Runs entirely on the local machine (no ROS, no Docker, no Stage).
Simulates a robot exploring a HouseExpo floor plan step by step:
1. Start at random position, take lidar scan
2. Build partial map from cumulative scans
3. Run diffusion model to predict complete map
4. Score frontiers by expected info gain
5. Move robot to best frontier
6. Repeat until coverage threshold

Produces:
- Frame-by-frame exploration video (MP4)
- Exploration GIF
- Coverage vs time plot
- Final results summary
"""

import os
import sys
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from diffusion import DDPMScheduler
from unet import ConditionalUNet

try:
    from PIL import Image
    import subprocess
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


def detect_frontier_clusters(known_mask, min_size=3):
    h, w = known_mask.shape
    frontiers = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
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


@torch.no_grad()
def score_frontiers_diffusion(model, scheduler, partial, known_mask, clusters,
                               robot_pos, device, K=4, info_radius=20,
                               lambda_var=1.0, beta_dist=0.3):
    pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    completions = []
    for _ in range(K):
        pred = scheduler.sample_ddim(model, pm, km, num_steps=30)
        completions.append((pred[0, 0].cpu().numpy() + 1) / 2)

    mean_completion = np.mean(completions, axis=0).clip(0, 1)
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
        dist = math.sqrt((cy - ry)**2 + (cx - rx)**2)
        score = expected + lambda_var * variance - beta_dist * dist
        scored.append((score, cy, cx, expected, variance))

    scored.sort(reverse=True)
    return scored, mean_completion


def score_frontiers_baseline(known_mask, clusters, robot_pos, info_radius=20, beta=0.3):
    h, w = known_mask.shape
    ry, rx = robot_pos
    scored = []
    for cy, cx, size in clusters:
        r = info_radius
        y1, y2 = max(0, cy-r), min(h, cy+r+1)
        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        unknown = float((known_mask[y1:y2, x1:x2] < 0.5).sum())
        dist = math.sqrt((cy - ry)**2 + (cx - rx)**2)
        score = unknown - beta * dist
        scored.append((score, cy, cx, unknown, 0.0))
    scored.sort(reverse=True)
    return scored


def simulate_exploration(model, scheduler, grid, device, out_dir,
                          max_steps=15, coverage_target=0.85,
                          mode="diffusion", K=4):
    """Run full exploration simulation."""
    h, w = grid.shape
    rng = np.random.default_rng(42)

    robot_pos = find_free_position(grid, rng)
    combined_visible = np.zeros((h, w), dtype=np.uint8)

    frames = []
    coverages = []
    ious_over_time = []
    step_times = []

    print(f"  Starting exploration ({mode} mode, max {max_steps} steps)...")

    for step in range(max_steps):
        t0 = time.time()

        visible = simulate_lidar(grid, robot_pos, num_rays=360, max_range_px=70)
        combined_visible = np.maximum(combined_visible, visible)

        partial = np.full_like(grid, 0.5)
        partial[combined_visible > 0] = grid[combined_visible > 0]
        known_mask = (combined_visible > 0).astype(np.float32)

        free_cells = (grid > 0.5).sum()
        known_free = ((grid > 0.5) & (combined_visible > 0)).sum()
        coverage = known_free / max(free_cells, 1)
        coverages.append(coverage)

        clusters = detect_frontier_clusters(known_mask)

        if not clusters or coverage >= coverage_target:
            print(f"    Step {step+1}: coverage={coverage:.0%} -- exploration complete!")

            fig = make_frame(grid, partial, known_mask, robot_pos, None,
                             [], step, coverage, 0, mode, None)
            frames.append(fig_to_frame(fig))
            plt.close(fig)
            break

        if mode == "diffusion":
            scored, mean_pred = score_frontiers_diffusion(
                model, scheduler, partial, known_mask, clusters,
                robot_pos, device, K=K)
            pred_iou = iou(grid, mean_pred)
            ious_over_time.append(pred_iou)
        else:
            scored = score_frontiers_baseline(known_mask, clusters, robot_pos)
            mean_pred = None
            pred_iou = 0
            ious_over_time.append(0)

        elapsed = time.time() - t0
        step_times.append(elapsed)

        best_score, best_y, best_x, best_gain, best_var = scored[0]

        fig = make_frame(grid, partial, known_mask, robot_pos, mean_pred,
                         scored, step, coverage, pred_iou, mode, coverages)
        frames.append(fig_to_frame(fig))
        plt.close(fig)

        print(f"    Step {step+1}: coverage={coverage:.0%} "
              f"{'IoU='+f'{pred_iou:.2f}' if mode=='diffusion' else ''} "
              f"gain={best_gain:.0f} ({elapsed:.1f}s)")

        # Move robot toward best frontier, scanning along the way
        ry, rx = robot_pos
        ty, tx = best_y, best_x
        dist = math.sqrt((ty-ry)**2 + (tx-rx)**2)
        n_intermediate = max(1, int(dist / 20))
        for si in range(1, n_intermediate + 1):
            frac = si / n_intermediate
            iy = int(ry + (ty - ry) * frac)
            ix = int(rx + (tx - rx) * frac)
            if 0 <= iy < h and 0 <= ix < w and grid[iy, ix] > 0.5:
                inter_vis = simulate_lidar(grid, (iy, ix), num_rays=360, max_range_px=70)
                combined_visible = np.maximum(combined_visible, inter_vis)
        robot_pos = (best_y, best_x)

    for _ in range(5):
        frames.append(frames[-1])

    gif_path = os.path.join(out_dir, f"exploration_{mode}.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=1500, loop=0)
    print(f"  GIF saved: {gif_path}")

    frame_dir = os.path.join(out_dir, f"frames_{mode}")
    os.makedirs(frame_dir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(frame_dir, f"frame_{i:03d}.png"))

    try:
        mp4_path = os.path.join(out_dir, f"exploration_{mode}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "1",
            "-i", os.path.join(frame_dir, "frame_%03d.png"),
            "-c:v", "libx264", "-profile:v", "baseline",
            "-pix_fmt", "yuv420p", "-r", "30",
            "-movflags", "+faststart",
            mp4_path
        ], capture_output=True, timeout=30)
        print(f"  MP4 saved: {mp4_path}")
    except Exception:
        print(f"  MP4 skipped (ffmpeg not available)")

    return coverages, ious_over_time, step_times


def make_frame(grid, partial, known_mask, robot_pos, mean_pred,
               scored, step, coverage, pred_iou, mode, coverage_history):
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(partial_rgb(partial))
    ax0.plot(robot_pos[1], robot_pos[0], "ro", markersize=10, markeredgecolor="black")

    if scored:
        for i, (s, sy, sx, g, v) in enumerate(scored[:5]):
            color = "lime" if i == 0 else "yellow" if i < 3 else "orange"
            size = 12 if i == 0 else 8
            marker = "*" if i == 0 else "o"
            ax0.plot(sx, sy, marker, color=color, markersize=size, markeredgecolor="black")

    ax0.set_title(f"Step {step+1}: Explored {coverage:.0%}\n"
                  f"{'*' if mode=='diffusion' else 'o'}=best frontier",
                  fontsize=11)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    if mean_pred is not None:
        ax1.imshow(mean_pred.clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
        ax1.set_title(f"Diffusion Prediction\n(IoU: {pred_iou:.2f})", fontsize=11)
    else:
        ax1.imshow(partial_rgb(partial))
        ax1.set_title("Baseline\n(no prediction)", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    ax2.plot(robot_pos[1], robot_pos[0], "ro", markersize=8)
    ax2.set_title("Ground Truth\n(robot doesn't see this)", fontsize=11)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[3])
    if coverage_history and len(coverage_history) > 1:
        ax3.plot(range(1, len(coverage_history)+1), coverage_history,
                 "o-", color="#4CAF50", linewidth=2, markersize=5)
        ax3.set_xlim(0, 16)
        ax3.set_ylim(0, 1)
        ax3.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Target 85%")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Coverage")
        ax3.set_title("Exploration Progress", fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, f"Coverage: {coverage:.0%}", ha="center", va="center",
                 fontsize=20, fontweight="bold", transform=ax3.transAxes)
        ax3.set_title("Exploration Progress", fontsize=11)
        ax3.axis("off")

    mode_label = "Diffusion-Guided" if mode == "diffusion" else "Baseline Heuristic"
    fig.suptitle(f"{mode_label} Exploration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def make_comparison_plot(diff_cov, base_cov, diff_ious, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(diff_cov)+1), diff_cov, "o-", color="#4CAF50",
                 linewidth=2, markersize=6, label="Diffusion")
    axes[0].plot(range(1, len(base_cov)+1), base_cov, "s-", color="#FF9800",
                 linewidth=2, markersize=6, label="Baseline")
    axes[0].axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Target")
    axes[0].set_xlabel("Exploration Step", fontsize=12)
    axes[0].set_ylabel("Coverage", fontsize=12)
    axes[0].set_title("Coverage Over Time", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    if diff_ious:
        axes[1].plot(range(1, len(diff_ious)+1), diff_ious, "o-", color="#2196F3",
                     linewidth=2, markersize=6)
        axes[1].set_xlabel("Exploration Step", fontsize=12)
        axes[1].set_ylabel("Prediction IoU", fontsize=12)
        axes[1].set_title("Diffusion Prediction Quality\nDuring Exploration", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coverage_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Coverage comparison saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/exploration_demo")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--map_index", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--K", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    plan = load_floor_plan(str(json_files[args.map_index]))
    grid = rasterize_floor_plan(plan, 256)
    print(f"Map {args.map_index}: {plan.get('room_num', '?')} rooms\n")

    print("=== Diffusion-guided exploration ===")
    diff_cov, diff_ious, diff_times = simulate_exploration(
        model, scheduler, grid, device, args.out_dir,
        max_steps=args.max_steps, mode="diffusion", K=args.K)

    print("\n=== Baseline exploration ===")
    base_cov, _, base_times = simulate_exploration(
        model, scheduler, grid, device, args.out_dir,
        max_steps=args.max_steps, mode="baseline")

    make_comparison_plot(diff_cov, base_cov, diff_ious, args.out_dir)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Diffusion: {diff_cov[-1]:.0%} coverage in {len(diff_cov)} steps "
          f"(avg {np.mean(diff_times):.1f}s/step)")
    print(f"Baseline:  {base_cov[-1]:.0%} coverage in {len(base_cov)} steps "
          f"(avg {np.mean(base_times):.1f}s/step)")
    if len(diff_cov) < len(base_cov):
        print(f"Diffusion reached target {len(base_cov) - len(diff_cov)} steps faster!")
    print(f"\nAll results saved to {args.out_dir}/")
