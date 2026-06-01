"""4-method synchronized GIF: same map, four robots, side-by-side animation.

Renders a 2x2 grid where each panel shows one frontier scorer running on the
same map from the same start, step by step. Output: animated GIF.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from simulate_exploration import detect_frontier_clusters
from experiments_4baseline import score_nearest, score_info_gain
from experiments import score_base, score_diff, load_model


JSON_DIR = Path("/Users/moin/Robotics-class/final-project/data/HouseExpo/HouseExpo/json")
JSON_FILES = sorted(JSON_DIR.glob("*.json"))
OUT = Path("results/analysis/figures/16_4method_sync.gif")
OUT.parent.mkdir(parents=True, exist_ok=True)

LIDAR = 70
MAX_STEPS = 12
SEED = 42

METHODS = ["nearest", "info_gain", "heuristic", "diffusion"]
LABELS = {
    "nearest": "Nearest",
    "info_gain": "Info-gain only",
    "heuristic": "Info-gain + dist",
    "diffusion": "Diffusion (ours)",
}
COLORS = {
    "nearest": "#aab1c0",
    "info_gain": "#7d6395",
    "heuristic": "#6fb6ff",
    "diffusion": "#c576ff",
}


def run_rollout_save_frames(grid, method, model=None, scheduler=None, device=None):
    """Returns list of (combined visibility array, robot_pos, coverage) per step."""
    h, w = grid.shape
    rng = np.random.default_rng(SEED)
    robot_pos = find_free_position(grid, rng)
    combined = np.zeros((h, w), dtype=np.uint8)
    frames = []

    for step in range(MAX_STEPS):
        vis = simulate_lidar(grid, robot_pos, num_rays=360, max_range_px=LIDAR)
        combined = np.maximum(combined, vis)
        known = (combined > 0).astype(np.float32)
        free = (grid > 0.5).sum()
        known_free = ((grid > 0.5) & (combined > 0)).sum()
        cov = float(known_free / max(free, 1))

        frames.append({"combined": combined.copy(), "robot": robot_pos, "cov": cov})

        if cov >= 0.95:
            break

        clusters = detect_frontier_clusters(known)
        clusters = [c for c in clusters
                    if math.hypot(c[0] - robot_pos[0], c[1] - robot_pos[1]) >= 10]
        if not clusters:
            break

        if method == "nearest":
            scored = score_nearest(clusters, robot_pos)
        elif method == "info_gain":
            scored = score_info_gain(known, clusters, robot_pos, info_radius=LIDAR)
        elif method == "heuristic":
            scored = score_base(known, clusters, robot_pos, info_radius=LIDAR)
        elif method == "diffusion":
            partial = np.full_like(grid, 0.5)
            partial[combined > 0] = grid[combined > 0]
            scored = score_diff(model, scheduler, partial, known, clusters, robot_pos,
                                device, info_radius=LIDAR)

        best = scored[0]
        ry, rx = robot_pos
        ty, tx = best[1], best[2]
        dist = math.sqrt((ty - ry) ** 2 + (tx - rx) ** 2)
        n_int = max(1, int(dist / 20))
        for si in range(1, n_int + 1):
            frac = si / n_int
            iy = int(ry + (ty - ry) * frac)
            ix = int(rx + (tx - rx) * frac)
            if 0 <= iy < h and 0 <= ix < w and grid[iy, ix] > 0.5:
                iv = simulate_lidar(grid, (iy, ix), num_rays=360, max_range_px=LIDAR)
                combined = np.maximum(combined, iv)
        robot_pos = (best[1], best[2])

    while len(frames) < MAX_STEPS:
        frames.append(frames[-1])
    return frames


def build(map_idx):
    plan = load_floor_plan(str(JSON_FILES[map_idx]))
    grid = rasterize_floor_plan(plan, 256)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model, scheduler = load_model("results/checkpoints/model_epoch0020.pt", device)

    print("Running rollouts...")
    method_frames = {}
    for m in METHODS:
        print(f"  {m}...")
        method_frames[m] = run_rollout_save_frames(
            grid, m, model=model, scheduler=scheduler, device=device,
        )

    n_frames = max(len(f) for f in method_frames.values())

    plt.rcParams.update({
        "figure.facecolor": "#0c1019",
        "axes.facecolor": "#0c1019",
        "text.color": "#dfe6f0",
    })

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=110, facecolor="#0c1019")
    axes = axes.flatten()
    title = fig.suptitle(
        f"Map {map_idx}: all four methods, same start, step 1",
        color="#dfe6f0", fontsize=14, fontweight="bold", y=0.96,
    )

    ims = {}
    cov_text = {}
    robot_pts = {}
    for ax, m in zip(axes, METHODS):
        ax.set_facecolor("#0c1019")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a3142")
        ax.set_title(LABELS[m], color=COLORS[m], fontsize=12, fontweight="bold", pad=8)

        img = np.zeros((*grid.shape, 3), dtype=np.float32)
        img[..., 0] = 0.06; img[..., 1] = 0.08; img[..., 2] = 0.13
        ims[m] = ax.imshow(img, interpolation="nearest", animated=True)
        robot_pts[m] = ax.scatter(
            [grid.shape[1] // 2], [grid.shape[0] // 2],
            s=40, color=COLORS[m], edgecolor="white", linewidth=1.0, zorder=5, animated=True,
        )
        cov_text[m] = ax.text(
            6, grid.shape[0] - 12, "0%",
            color=COLORS[m], fontsize=14, fontweight="bold",
            bbox=dict(facecolor="#0c1019", edgecolor="none", pad=2, alpha=0.7),
            animated=True,
        )

    def update(frame_idx):
        title.set_text(
            f"Map {map_idx}: all four methods, same start, step {frame_idx + 1}"
        )
        artists = [title]
        for m in METHODS:
            f = method_frames[m][min(frame_idx, len(method_frames[m]) - 1)]
            img = np.zeros((*grid.shape, 3), dtype=np.float32)
            img[..., 0] = 0.06; img[..., 1] = 0.08; img[..., 2] = 0.13
            combined = f["combined"]
            known_free = (combined > 0) & (grid > 0.5)
            known_wall = (combined > 0) & (grid <= 0.5)
            img[known_free] = (0.94, 0.96, 0.99)
            img[known_wall] = (0.22, 0.28, 0.4)
            ims[m].set_array(img)
            ry, rx = f["robot"]
            robot_pts[m].set_offsets([[rx, ry]])
            cov_text[m].set_text(f"{f['cov'] * 100:.0f}%")
            artists.extend([ims[m], robot_pts[m], cov_text[m]])
        return artists

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    anim = FuncAnimation(fig, update, frames=n_frames, interval=600, blit=False)
    print(f"Writing {OUT}...")
    writer = PillowWriter(fps=2)
    anim.save(str(OUT), writer=writer, dpi=110)
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build(map_idx=2638)
