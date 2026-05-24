"""Pre-compute diffusion-guided exploration waypoints on the actual Stage maze.

Loads maze.png (the Stage world), simulates exploration using the diffusion
model, and outputs a sequence of (x, y) waypoints in Stage world coordinates.

The robot in Stage starts at pose (2.0, 2.0). The maze is 10m x 10m.
maze.png is 200x200 pixels, so 1 pixel = 0.05m.
"""

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import find_free_position, simulate_lidar
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


def load_maze(maze_png_path, target_size=256):
    img = np.array(Image.open(maze_png_path).convert("L"))
    grid = (img > 128).astype(np.float32)
    if grid.shape[0] != target_size:
        from PIL import Image as PILImage
        resized = PILImage.fromarray((grid * 255).astype(np.uint8)).resize(
            (target_size, target_size), PILImage.NEAREST)
        grid = (np.array(resized) > 128).astype(np.float32)
    return grid


def pixel_to_world(py, px, grid_size=256, world_size=10.0):
    wx = px / grid_size * world_size
    wy = (grid_size - py) / grid_size * world_size
    return wx, wy


def world_to_pixel(wx, wy, grid_size=256, world_size=10.0):
    px = int(wx / world_size * grid_size)
    py = int((world_size - wy) / world_size * grid_size)
    return py, px


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
def score_frontiers(model, scheduler, partial, known_mask, clusters,
                    robot_pos, device, K=4):
    pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    completions = []
    for _ in range(K):
        pred = scheduler.sample_ddim(model, pm, km, num_steps=30)
        completions.append((pred[0, 0].cpu().numpy() + 1) / 2)

    ry, rx = robot_pos
    scored = []
    for cy, cx, size in clusters:
        gains = []
        r = 20
        h, w = known_mask.shape
        for comp in completions:
            y1, y2 = max(0, cy-r), min(h, cy+r+1)
            x1, x2 = max(0, cx-r), min(w, cx+r+1)
            gain = float(((comp[y1:y2, x1:x2] > 0.5) & (known_mask[y1:y2, x1:x2] < 0.5)).sum())
            gains.append(gain)
        expected = np.mean(gains)
        variance = np.std(gains)
        dist = math.sqrt((cy - ry)**2 + (cx - rx)**2)
        score = expected + 1.0 * variance - 0.3 * dist
        scored.append((score, cy, cx))
    scored.sort(reverse=True)
    return scored, np.mean(completions, axis=0).clip(0, 1)


def partial_rgb(p):
    vis = np.zeros((*p.shape, 3))
    vis[p > 0.6] = [1, 1, 1]
    vis[p < 0.4] = [0, 0, 0]
    vis[(p >= 0.4) & (p <= 0.6)] = [0.5, 0.5, 0.7]
    return vis


def precompute(model, scheduler, grid, device, out_dir,
               start_world=(2.0, 2.0), max_steps=20, K=4):
    h, w = grid.shape
    robot_py, robot_px = world_to_pixel(*start_world)
    robot_pos = (robot_py, robot_px)

    combined_visible = np.zeros((h, w), dtype=np.uint8)
    waypoints = [{"step": 0, "world_x": start_world[0], "world_y": start_world[1],
                  "pixel_y": robot_py, "pixel_x": robot_px}]

    frames = []

    for step in range(max_steps):
        visible = simulate_lidar(grid, robot_pos, num_rays=360, max_range_px=80)
        combined_visible = np.maximum(combined_visible, visible)

        partial = np.full_like(grid, 0.5)
        partial[combined_visible > 0] = grid[combined_visible > 0]
        known_mask = (combined_visible > 0).astype(np.float32)

        free_cells = (grid > 0.5).sum()
        known_free = ((grid > 0.5) & (combined_visible > 0)).sum()
        coverage = known_free / max(free_cells, 1)

        clusters = detect_frontier_clusters(known_mask)
        if not clusters or coverage >= 0.85:
            print(f"  Step {step+1}: coverage={coverage:.0%} -- done!")
            break

        scored, mean_pred = score_frontiers(
            model, scheduler, partial, known_mask, clusters, robot_pos, device, K=K)

        best_score, best_y, best_x = scored[0]
        wx, wy = pixel_to_world(best_y, best_x)

        waypoints.append({
            "step": step + 1,
            "world_x": round(wx, 3),
            "world_y": round(wy, 3),
            "pixel_y": best_y,
            "pixel_x": best_x,
            "score": round(best_score, 1),
            "coverage": round(float(coverage), 3),
        })

        print(f"  Step {step+1}: coverage={coverage:.0%} → go to ({wx:.1f}, {wy:.1f})")

        # Make frame
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(partial_rgb(partial))
        axes[0].plot(robot_pos[1], robot_pos[0], "ro", markersize=10)
        for i, (s, sy, sx) in enumerate(scored[:5]):
            color = "lime" if i == 0 else "yellow"
            axes[0].plot(sx, sy, "*" if i == 0 else "o", color=color, markersize=12 if i == 0 else 6)
        axes[0].set_title(f"Step {step+1}: {coverage:.0%} explored")
        axes[0].axis("off")

        axes[1].imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title("Diffusion Prediction")
        axes[1].axis("off")

        axes[2].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[2].plot(robot_pos[1], robot_pos[0], "ro", markersize=8)
        axes[2].plot(best_x, best_y, "g*", markersize=14)
        axes[2].set_title("Ground Truth + Goal")
        axes[2].axis("off")

        plt.suptitle("Diffusion-Guided Frontier Selection on Stage Maze", fontweight="bold")
        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        fw, fh = fig.canvas.get_width_height()
        frame = Image.fromarray(np.asarray(buf).reshape(fh, fw, 4)[:, :, :3].copy())
        frames.append(frame)
        plt.close()

        # Move robot, scanning along the way
        ry, rx = robot_pos
        dist = math.sqrt((best_y - ry)**2 + (best_x - rx)**2)
        n_inter = max(1, int(dist / 15))
        for si in range(1, n_inter + 1):
            frac = si / n_inter
            iy = int(ry + (best_y - ry) * frac)
            ix = int(rx + (best_x - rx) * frac)
            if 0 <= iy < h and 0 <= ix < w and grid[iy, ix] > 0.5:
                v = simulate_lidar(grid, (iy, ix), num_rays=360, max_range_px=80)
                combined_visible = np.maximum(combined_visible, v)
        robot_pos = (best_y, best_x)

    # Save waypoints
    wp_path = os.path.join(out_dir, "waypoints.json")
    with open(wp_path, "w") as f:
        json.dump(waypoints, f, indent=2)
    print(f"\n  Waypoints saved: {wp_path}")

    # Save GIF
    for _ in range(5):
        frames.append(frames[-1])
    gif_path = os.path.join(out_dir, "maze_exploration.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1500, loop=0)
    print(f"  GIF saved: {gif_path}")

    return waypoints


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--maze_png", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/stage_demo")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--K", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)

    print("Loading maze...")
    grid = load_maze(args.maze_png)
    print(f"  Maze: {grid.shape}, {(grid > 0.5).sum()} free cells\n")

    print("Computing exploration waypoints...")
    waypoints = precompute(model, scheduler, grid, device, args.out_dir, K=args.K)

    print(f"\n{len(waypoints)} waypoints computed. Ready for Stage replay.")
