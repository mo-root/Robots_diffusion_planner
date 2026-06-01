"""Visual side-by-side rollout: 4 methods on the same map at the same steps.

Picks two maps with dramatic differences (one big diffusion win, one near-tie),
renders each method's partial map at steps {1, 2, 4, 8, 12, final}, and writes
a single composite figure per map.
"""

import math
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from simulate_exploration import detect_frontier_clusters
from experiments_4baseline import score_nearest, score_info_gain
from experiments import score_base, score_diff, load_model


JSON_DIR = Path("/Users/moin/Robotics-class/final-project/data/HouseExpo/HouseExpo/json")
OUT_DIR = Path("results/analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIDAR = 70
MAX_STEPS = 20
STEPS_TO_SHOW = [1, 2, 4, 8, 12, 20]
SEED = 42

METHOD_ORDER = ["nearest", "info_gain", "heuristic", "diffusion"]
METHOD_LABELS = {
    "nearest": "Nearest",
    "info_gain": "Info-gain only",
    "heuristic": "Info-gain + dist",
    "diffusion": "Diffusion (ours)",
}
METHOD_COLOR = {
    "nearest": "#aab1c0",
    "info_gain": "#7d6395",
    "heuristic": "#6fb6ff",
    "diffusion": "#c576ff",
}


def rollout_capture(grid, method, snapshot_steps, model=None, scheduler=None, device=None):
    """Run a real rollout for `method` and snapshot the partial map at the given steps."""
    h, w = grid.shape
    rng = np.random.default_rng(SEED)
    robot_pos = find_free_position(grid, rng)
    combined = np.zeros((h, w), dtype=np.uint8)

    snapshots = {}
    for step in range(MAX_STEPS):
        vis = simulate_lidar(grid, robot_pos, num_rays=360, max_range_px=LIDAR)
        combined = np.maximum(combined, vis)
        known = (combined > 0).astype(np.float32)
        free = (grid > 0.5).sum()
        known_free = ((grid > 0.5) & (combined > 0)).sum()
        cov = float(known_free / max(free, 1))

        if (step + 1) in snapshot_steps:
            snapshots[step + 1] = {
                "combined": combined.copy(),
                "robot": robot_pos,
                "cov": cov,
            }

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
        else:
            raise ValueError(method)

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

    last_step = max(snapshots) if snapshots else 0
    for s in snapshot_steps:
        if s not in snapshots and last_step:
            snapshots[s] = dict(snapshots[last_step])
    return snapshots


JSON_FILES = sorted(JSON_DIR.glob("*.json"))


def build_panel(map_idx, model=None, scheduler=None, device=None):
    plan = load_floor_plan(str(JSON_FILES[map_idx]))
    grid = rasterize_floor_plan(plan, 256)

    panels = {
        m: rollout_capture(grid, m, STEPS_TO_SHOW,
                           model=model, scheduler=scheduler, device=device)
        for m in METHOD_ORDER
    }

    fig, axes = plt.subplots(
        len(METHOD_ORDER), len(STEPS_TO_SHOW),
        figsize=(2.4 * len(STEPS_TO_SHOW), 2.4 * len(METHOD_ORDER) + 0.6),
        dpi=130,
        facecolor="#0c1019",
    )

    for r, m in enumerate(METHOD_ORDER):
        for c, step in enumerate(STEPS_TO_SHOW):
            ax = axes[r, c]
            ax.set_facecolor("#0c1019")
            snap = panels[m].get(step)
            if snap is None:
                ax.axis("off")
                continue
            combined = snap["combined"]
            ry, rx = snap["robot"]
            cov = snap["cov"]

            img = np.zeros((*grid.shape, 3), dtype=np.float32)
            img[..., 0] = 0.06; img[..., 1] = 0.08; img[..., 2] = 0.13  # unknown bg
            known_free = (combined > 0) & (grid > 0.5)
            known_wall = (combined > 0) & (grid <= 0.5)
            img[known_free] = (0.94, 0.96, 0.99)  # white free
            img[known_wall] = (0.22, 0.28, 0.4)   # walls

            ax.imshow(img, interpolation="nearest")
            ax.scatter([rx], [ry], s=22,
                       color=METHOD_COLOR[m], edgecolor="white", linewidth=0.8, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#2a3142")

            if r == 0:
                title = f"step {step}" if step < MAX_STEPS else "final"
                ax.set_title(title, color="#dfe6f0", fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(
                    METHOD_LABELS[m], color=METHOD_COLOR[m],
                    fontsize=12, fontweight="bold", labelpad=10, rotation=90,
                )
            ax.text(
                4, grid.shape[0] - 8, f"{cov * 100:.0f}%",
                color=METHOD_COLOR[m], fontsize=11, fontweight="bold",
                bbox=dict(facecolor="#0c1019", edgecolor="none", pad=2, alpha=0.7),
            )

    fig.suptitle(
        f"Map {map_idx}: 4-method rollout from the same start position",
        color="#dfe6f0", fontsize=14, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = OUT_DIR / f"14_rollout_map{map_idx}.png"
    fig.savefig(out_path, dpi=130, facecolor="#0c1019")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading diffusion model on {device}...")
    model, scheduler = load_model("results/checkpoints/model_epoch0020.pt", device)
    # 2638 and 17666: big diffusion wins; 17925: near-tie + honest failure case
    for m in [2638, 17666, 17925]:
        try:
            build_panel(m, model=model, scheduler=scheduler, device=device)
        except Exception as e:
            print(f"map {m}: {e}")
