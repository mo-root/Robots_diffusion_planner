"""Behind-the-mind composite: world / partial map / diffusion imagination, per step.

Three rows, N columns (one per snapshot step):
    row 1: ground truth + robot trail (what is real)
    row 2: the partial occupancy grid the robot has accumulated (what the robot knows)
    row 3: mean of K=8 diffusion completions with the chosen frontier marked
            (what the model imagines and picks)

Goal: make the closed loop visible -- world -> observation -> imagination -> decision.
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
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from diffusion import DDPMScheduler
from unet import ConditionalUNet
from simulate_exploration import detect_frontier_clusters
from experiments import score_diff, load_model


JSON_DIR = Path("/Users/moin/Robotics-class/final-project/data/HouseExpo/HouseExpo/json")
JSON_FILES = sorted(JSON_DIR.glob("*.json"))
OUT_DIR = Path("results/analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIDAR = 70
MAX_STEPS = 20
SNAPSHOT_STEPS = [1, 2, 4, 8]
SEED = 42
K = 4  # smaller for cleaner mean visualization


@torch.no_grad()
def diffusion_completions(model, scheduler, partial, known_mask, device, K=K):
    pm = torch.tensor(partial * 2 - 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    comps = []
    for _ in range(K):
        pred = scheduler.sample_ddim(model, pm, km, num_steps=30)
        comps.append((pred[0, 0].cpu().numpy() + 1) / 2)
    return np.stack(comps)


def render(map_idx, model, scheduler, device):
    plan = load_floor_plan(str(JSON_FILES[map_idx]))
    grid = rasterize_floor_plan(plan, 256)
    h, w = grid.shape
    rng = np.random.default_rng(SEED)
    robot_pos = find_free_position(grid, rng)

    combined = np.zeros((h, w), dtype=np.uint8)
    trail = [robot_pos]
    snapshots = {}

    for step in range(MAX_STEPS):
        vis = simulate_lidar(grid, robot_pos, num_rays=360, max_range_px=LIDAR)
        combined = np.maximum(combined, vis)
        known = (combined > 0).astype(np.float32)
        partial = np.full_like(grid, 0.5)
        partial[combined > 0] = grid[combined > 0]

        if (step + 1) in SNAPSHOT_STEPS:
            clusters_for_view = detect_frontier_clusters(known)
            clusters_for_view = [c for c in clusters_for_view
                                 if math.hypot(c[0] - robot_pos[0], c[1] - robot_pos[1]) >= 10]

            comps = diffusion_completions(model, scheduler, partial, known, device)
            mean_comp = comps.mean(axis=0)

            target = None
            if clusters_for_view:
                scored = score_diff(model, scheduler, partial, known, clusters_for_view,
                                    robot_pos, device, info_radius=LIDAR)
                target = (scored[0][1], scored[0][2])

            snapshots[step + 1] = {
                "combined": combined.copy(),
                "partial": partial.copy(),
                "known": known.copy(),
                "robot": robot_pos,
                "trail": list(trail),
                "mean_comp": mean_comp,
                "frontiers": clusters_for_view,
                "target": target,
            }

        free = (grid > 0.5).sum()
        known_free = ((grid > 0.5) & (combined > 0)).sum()
        cov = float(known_free / max(free, 1))
        if cov >= 0.95:
            for s in SNAPSHOT_STEPS:
                if s not in snapshots:
                    snapshots[s] = snapshots[max(snapshots)]
            break

        clusters = detect_frontier_clusters(known)
        clusters = [c for c in clusters
                    if math.hypot(c[0] - robot_pos[0], c[1] - robot_pos[1]) >= 10]
        if not clusters:
            break

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
            trail.append((iy, ix))
        robot_pos = (best[1], best[2])
        trail.append(robot_pos)

    return grid, snapshots


def plot(map_idx, grid, snapshots, label):
    cols = len(SNAPSHOT_STEPS)
    fig, axes = plt.subplots(
        3, cols,
        figsize=(2.7 * cols, 2.7 * 3 + 0.8),
        dpi=130,
        facecolor="#0c1019",
    )

    row_labels = [
        ("The world", "ground truth + robot trail"),
        ("The robot's view", "partial occupancy + frontiers"),
        ("The model's mind", "mean of K=4 diffusion samples"),
    ]
    for r, (rlbl, rsub) in enumerate(row_labels):
        axes[r, 0].set_ylabel(
            rlbl, rotation=90, fontsize=13, fontweight="bold",
            color="#dfe6f0", labelpad=14,
        )

    for c, step in enumerate(SNAPSHOT_STEPS):
        snap = snapshots.get(step)
        if snap is None:
            for r in range(3):
                axes[r, c].axis("off")
            continue

        ax = axes[0, c]
        img = np.zeros((*grid.shape, 3), dtype=np.float32)
        img[grid > 0.5] = (0.94, 0.96, 0.99)
        img[grid <= 0.5] = (0.22, 0.28, 0.4)
        ax.imshow(img, interpolation="nearest")
        if snap["trail"]:
            ys = [p[0] for p in snap["trail"]]
            xs = [p[1] for p in snap["trail"]]
            ax.plot(xs, ys, color="#ffb86c", linewidth=1.5, alpha=0.9)
        ry, rx = snap["robot"]
        ax.scatter([rx], [ry], s=42, color="#ff7b72",
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.set_title(f"step {step}", color="#dfe6f0", fontsize=12, pad=6)

        ax = axes[1, c]
        img = np.full((*grid.shape, 3), [0.05, 0.07, 0.13], dtype=np.float32)
        combined = snap["combined"]
        known_free = (combined > 0) & (grid > 0.5)
        known_wall = (combined > 0) & (grid <= 0.5)
        img[known_free] = (0.94, 0.96, 0.99)
        img[known_wall] = (0.22, 0.28, 0.4)
        ax.imshow(img, interpolation="nearest")
        for cy, cx, _ in snap["frontiers"]:
            ax.scatter([cx], [cy], s=18, color="#6fb6ff",
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter([rx], [ry], s=42, color="#ff7b72",
                   edgecolor="white", linewidth=1.2, zorder=5)

        ax = axes[2, c]
        mean_comp = snap["mean_comp"]
        cmap_img = np.zeros((*grid.shape, 3), dtype=np.float32)
        cmap_img[..., 0] = 0.06 + 0.6 * mean_comp
        cmap_img[..., 1] = 0.05 + 0.4 * mean_comp
        cmap_img[..., 2] = 0.13 + 0.7 * mean_comp
        ax.imshow(cmap_img, interpolation="nearest")
        if snap["target"] is not None:
            ty, tx = snap["target"]
            circ = Circle((tx, ty), 12, edgecolor="#c576ff",
                          facecolor="none", linewidth=2.4, zorder=5)
            ax.add_patch(circ)
            ax.annotate(
                "pick", xy=(tx, ty), xytext=(tx + 20, ty - 20),
                color="#c576ff", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c576ff", lw=1.4),
            )
        ax.scatter([rx], [ry], s=42, color="#ff7b72",
                   edgecolor="white", linewidth=1.2, zorder=5)

        for r in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_color("#2a3142")
            axes[r, c].set_facecolor("#0c1019")

    fig.suptitle(
        f"Behind the mind  |  map {map_idx} ({label})",
        color="#dfe6f0", fontsize=15, y=0.995, fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = OUT_DIR / f"15_behind_mind_map{map_idx}.png"
    fig.savefig(out, dpi=130, facecolor="#0c1019")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading diffusion model on {device}...")
    model, scheduler = load_model("results/checkpoints/model_epoch0020.pt", device)

    for m, label in [(2638, "diffusion wins +23pp at step 4"),
                     (17666, "diffusion wins -- branching apartment")]:
        try:
            grid, snaps = render(m, model, scheduler, device)
            plot(m, grid, snaps, label)
        except Exception as e:
            print(f"map {m}: {e}")
            import traceback; traceback.print_exc()
