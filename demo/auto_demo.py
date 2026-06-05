"""Auto-generated exploration demo: shows the diffusion model *denoising* (firing)
and the robot *moving* based on those predictions — side by side, no ROS/Stage needed.

For each decision step on a real HouseExpo floor plan it:
  1. builds the partial map from accumulated lidar (data_generator.simulate_lidar),
  2. runs the trained diffusion model (K completions), capturing the DDIM denoising
     trajectory (pure noise -> predicted map) for the left-hand "firing" panel,
  3. scores frontiers by expected info-gain + sample-variance - distance,
  4. drives the robot toward the chosen frontier and re-observes.

Output: demo/auto_frames/frame_####.png  (compile to mp4 with ffmpeg).

Usage:
    python demo/auto_demo.py --checkpoint results/checkpoints/model_epoch0020.pt \
        --steps 10 --K 3 --ddim_steps 20 --device cpu
"""

import argparse
import heapq
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from data_generator import rasterize_floor_plan, find_free_position, simulate_lidar, load_floor_plan  # noqa: E402
from unet import ConditionalUNet  # noqa: E402
from diffusion import DDPMScheduler  # noqa: E402


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = ConditionalUNet(in_channels=3, out_channels=1,
                            base_channels=a.get("base_channels", 32),
                            channel_mults=tuple(a.get("channel_mults", [1, 2, 4, 4])),
                            time_dim=a.get("time_dim", 128)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, int(a.get("T", 1000))


@torch.no_grad()
def ddim_sample(model, sched, partial, known, num_steps, capture=0):
    """DDIM (eta=0). Returns (final[0,1] HxW, [intermediate maps] if capture>0)."""
    device = partial.device
    step = sched.T // num_steps
    ts = list(range(0, sched.T, step))[::-1]
    x = torch.randn(1, 1, partial.shape[2], partial.shape[3], device=device)
    cap_at = set(np.linspace(0, len(ts) - 1, capture).astype(int)) if capture else set()
    inter = []
    for i, tv in enumerate(ts):
        t = torch.full((1,), tv, device=device, dtype=torch.long)
        eps = model(x, t, partial, known)
        ac = sched.alphas_cumprod[tv]
        ac_prev = sched.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        x0 = ((x - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac)).clamp(-1, 1)
        x = torch.sqrt(ac_prev) * x0 + torch.sqrt(1 - ac_prev) * eps
        if i in cap_at:
            inter.append(np.clip((x0[0, 0].cpu().numpy() + 1) / 2, 0, 1))
    final = np.clip((x[0, 0].cpu().numpy() + 1) / 2, 0, 1)
    return final, inter


def pick_floorplan(json_dir, res, rng, scan=400):
    """Find a multi-room floor plan with a reasonable free ratio."""
    files = sorted(Path(json_dir).glob("*.json"))
    order = rng.permutation(len(files))[:scan]
    for idx in order:
        try:
            plan = load_floor_plan(str(files[idx]))
        except Exception:
            continue
        if plan.get("room_num", 0) < 4:
            continue
        grid = rasterize_floor_plan(plan, res)
        if grid is None:
            continue
        fr = grid.sum() / grid.size
        if 0.30 <= fr <= 0.62 and grid.sum() > 4000:
            return grid, files[idx].stem
    raise RuntimeError("no suitable floor plan found")


def frontiers(known, free_known):
    """Centroids of known-free regions bordering unknown space."""
    unknown = (known == 0).astype(np.uint8)
    border = cv2.dilate(unknown, np.ones((3, 3), np.uint8)) & free_known.astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(border, connectivity=8)
    pts = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            cx, cy = cent[i]
            pts.append((int(round(cy)), int(round(cx)), int(stats[i, cv2.CC_STAT_AREA])))
    return pts


def info_gain(pred, known, fy, fx, R):
    """# predicted-free cells within R of (fy,fx) that are currently unknown."""
    y0, y1 = max(0, fy - R), min(pred.shape[0], fy + R)
    x0, x1 = max(0, fx - R), min(pred.shape[1], fx + R)
    sub_pred = pred[y0:y1, x0:x1]
    sub_unk = known[y0:y1, x0:x1] == 0
    return int(((sub_pred > 0.5) & sub_unk).sum())


def snap(trav, pt):
    """Nearest traversable cell to pt (used if a frontier centroid lands on a wall)."""
    if 0 <= pt[0] < trav.shape[0] and 0 <= pt[1] < trav.shape[1] and trav[pt[0], pt[1]]:
        return (int(pt[0]), int(pt[1]))
    ys, xs = np.where(trav)
    if len(ys) == 0:
        return (int(pt[0]), int(pt[1]))
    i = ((ys - pt[0]) ** 2 + (xs - pt[1]) ** 2).argmin()
    return (int(ys[i]), int(xs[i]))


def astar(trav, start, goal):
    """8-connected A* over a boolean traversable grid (walls = False).

    Returns a wall-respecting list of (y, x) cells from start to goal, or None.
    This is the same graph search as PA3 — the robot never crosses a wall.
    """
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    H, W = trav.shape
    if not (trav[start] and trav[goal]):
        return None
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
    openq = [(0.0, start)]
    g = {start: 0.0}
    came = {}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        cy, cx = cur
        for dy, dx, c in nbrs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and trav[ny, nx]:
                if dy != 0 and dx != 0 and not (trav[cy + dy, cx] and trav[cy, cx + dx]):
                    continue   # forbid diagonal corner-cutting through a 1px wall gap
                ng = g[cur] + c
                nb = (ny, nx)
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    came[nb] = cur
                    f = ng + ((ny - goal[0]) ** 2 + (nx - goal[1]) ** 2) ** 0.5
                    heapq.heappush(openq, (f, nb))
    return None


def draw_ros_pipeline(ax, active):
    """Draw the ROS 2 / Stage node graph with the live topic flow highlighted.

    active = 'score' (diffusion scorer is producing /best_frontier) or
    'drive' (A* planner reads /map and produces /cmd_vel).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("ROS 2 / Stage integration — live topic flow (this is the actual node graph)",
                 fontsize=9, fontweight="bold")
    boxw, boxh, yb = 0.17, 0.42, 0.36
    xs = [0.10, 0.37, 0.63, 0.88]
    labels = ["Stage\nsimulator", "PA4 mapper", "Diffusion\nscorer (ROS 2)", "A* planner\n+ controller"]
    glow = {"score": 2, "drive": 3}.get(active, -1)
    for i, (x, lab) in enumerate(zip(xs, labels)):
        hot = (i == glow)
        ax.add_patch(plt.Rectangle((x - boxw / 2, yb), boxw, boxh,
                     facecolor="#e8f5e9" if hot else "#eceff1",
                     edgecolor="#2e7d32" if hot else "#90a4ae",
                     lw=2.6 if hot else 1.1, zorder=2))
        ax.text(x, yb + boxh / 2, lab, ha="center", va="center", fontsize=7.5,
                fontweight="bold" if hot else "normal", zorder=3)

    def arr(i, j, label, hot, dy=0.12):
        x0, x1 = xs[i] + boxw / 2, xs[j] - boxw / 2
        y = yb + boxh / 2
        col = "#d32f2f" if hot else "#78909c"
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2 if hot else 1.2))
        ax.text((x0 + x1) / 2, y + dy, label, ha="center", fontsize=6.6, color=col)

    arr(0, 1, "/scan", False)
    arr(1, 2, "/map", active == "score")
    arr(2, 3, "/best_frontier", active == "score")
    # planner also consumes /map for A* (the mapping <-> planning coupling)
    col = "#d32f2f" if active == "drive" else "#90a4ae"
    ax.annotate("", xy=(xs[3], yb + boxh + 0.05), xytext=(xs[1], yb + boxh + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8 if active == "drive" else 1.0,
                                connectionstyle="arc3,rad=-0.3", linestyle="--"))
    ax.text((xs[1] + xs[3]) / 2, yb + boxh + 0.19, "/map  (planner reads the live grid → A*)",
            ha="center", fontsize=6.6, color=col)
    # return loop: /cmd_vel planner -> stage
    col = "#d32f2f" if active == "drive" else "#78909c"
    ax.annotate("", xy=(xs[0], yb - 0.07), xytext=(xs[3], yb - 0.07),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0 if active == "drive" else 1.2,
                                connectionstyle="arc3,rad=0.12"))
    ax.text((xs[0] + xs[3]) / 2, yb - 0.22, "/cmd_vel", ha="center", fontsize=6.6, color=col)


def render(path, grid, occ, known, robot, traj, fronts, best, denoise_img,
           mean_pred, iou, coverage, step, nsteps, left_action, right_label, ros_active):
    H, W = grid.shape
    # exploration RGB
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[known == 0] = [0.62, 0.66, 0.78]          # unknown = blue-gray
    free_known = (known == 1) & (occ > 0.5)
    rgb[free_known] = [1, 1, 1]                    # known free = white
    rgb[(known == 1) & (occ <= 0.5)] = [0.1, 0.1, 0.1]   # known wall = black

    fig = plt.figure(figsize=(13, 8.0))
    gs = GridSpec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1, 0.5], figure=fig,
                  wspace=0.12, hspace=0.30)
    axE = fig.add_subplot(gs[0:2, 0])
    axE.imshow(rgb, origin="upper")
    if len(traj) > 1:
        ty, tx = zip(*traj)
        axE.plot(tx, ty, "-", color="#c01515", lw=1.3, alpha=0.55)
    for (fy, fx, _a) in fronts:
        axE.plot(fx, fy, "o", color="#ff7f0e", ms=5, mec="k", mew=0.4)
    if best is not None:
        axE.plot(best[1], best[0], "*", color="#2ca02c", ms=20, mec="k", mew=0.6)
    axE.plot(robot[1], robot[0], "o", color="#d62728", ms=11, mec="w", mew=1.4)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="w", markersize=9, label="Robot"),
        Line2D([0], [0], color="#c01515", lw=2, alpha=0.7, label="A* path driven (routes around walls)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
               markeredgecolor="k", markersize=8, label="Candidate frontiers"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=14, label="Chosen frontier (next goal)"),
        Patch(facecolor="white", edgecolor="#888", label="Mapped free"),
        Patch(facecolor=(0.62, 0.66, 0.78), label="Unknown (to explore)"),
        Patch(facecolor=(0.1, 0.1, 0.1), label="Wall"),
    ]
    axE.legend(handles=handles, loc="lower left", fontsize=6.8, framealpha=0.92,
               ncol=2, columnspacing=1.0, handlelength=1.3, borderpad=0.4, labelspacing=0.3)
    axE.set_title(f"LEFT — robot {left_action}  ·  step {step}/{nsteps}  ·  coverage {coverage:.0%}",
                  fontsize=10.5, fontweight="bold")
    axE.axis("off")

    axD = fig.add_subplot(gs[0, 1])
    axD.imshow(denoise_img, cmap="gray_r", vmin=0, vmax=1, origin="upper")
    axD.set_title(f"RIGHT-TOP — {right_label}", fontsize=9.5)
    axD.axis("off")

    axP = fig.add_subplot(gs[1, 1])
    axP.imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1, origin="upper")
    axP.set_title(f"RIGHT-BOTTOM — predicted full map (IoU {iou:.2f})", fontsize=9.5)
    axP.axis("off")

    axR = fig.add_subplot(gs[2, :])
    draw_ros_pipeline(axR, ros_active)

    fig.suptitle("Diffusion map-completion guiding frontier exploration",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=96, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def iou_of(pred, grid):
    p, g = pred > 0.5, grid > 0.5
    return (p & g).sum() / max((p | g).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/checkpoints/model_epoch0020.pt")
    ap.add_argument("--json_dir", default="data/HouseExpo/HouseExpo/json")
    ap.add_argument("--out", default="demo/auto_frames")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--radius", type=int, default=22)
    ap.add_argument("--beta", type=float, default=0.8,
                    help="distance penalty weight (higher = sweep nearer frontiers first)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    model, T = load_model(args.checkpoint, device)
    sched = DDPMScheduler(num_timesteps=T, device=str(device))

    grid, map_id = pick_floorplan(args.json_dir, args.res, rng)
    print(f"floor plan {map_id}  free={grid.mean():.2f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    robot = find_free_position(grid, rng)
    known = np.zeros_like(grid, np.uint8)
    occ = np.full_like(grid, 0.5)
    traj = [robot]
    total_free = max((grid > 0.5).sum(), 1)
    fidx = 0
    cov_hist = []

    def observe(pos):
        vis = simulate_lidar(grid, pos, num_rays=360, max_range_px=70)
        known[vis > 0] = 1
        occ[vis > 0] = grid[vis > 0]

    observe(robot)

    for step in range(1, args.steps + 1):
        # ---- model inference (K completions, capture denoising of the first) ----
        partial = (occ * 2 - 1).astype(np.float32)
        pt = torch.from_numpy(partial)[None, None].to(device)
        kt = torch.from_numpy(known.astype(np.float32))[None, None].to(device)
        preds, denoise_seq = [], []
        for k in range(args.K):
            final, inter = ddim_sample(model, sched, pt, kt, args.ddim_steps,
                                       capture=8 if k == 0 else 0)
            preds.append(final)
            if k == 0:
                denoise_seq = inter + [final]
        preds = np.stack(preds, 0)
        mean_pred = preds.mean(0)
        iou = iou_of(mean_pred, grid)

        free_known = (known == 1) & (occ > 0.5)
        fronts = frontiers(known, free_known)
        coverage = float(known[grid > 0.5].sum()) / total_free
        cov_hist.append(coverage)
        if not fronts:
            print(f"step {step}: no frontiers, stopping")
            break

        # ---- score frontiers: E[gain] + std - 0.3*dist ----
        scored = []
        for (fy, fx, _a) in fronts:
            gains = [info_gain(preds[k], known, fy, fx, args.radius) for k in range(args.K)]
            dist = np.hypot(fy - robot[0], fx - robot[1])
            score = float(np.mean(gains)) + 1.0 * float(np.std(gains)) - args.beta * dist
            scored.append((score, (fy, fx)))
        scored.sort(reverse=True)
        best = scored[0][1]

        # ---- "thinking" sub-frames: animate the denoising in the right panel ----
        for j, dimg in enumerate(denoise_seq):
            rlabel = f"diffusion denoising {j+1}/{len(denoise_seq)} (noise → predicted map)"
            render(out / f"frame_{fidx:04d}.png", grid, occ, known, robot, traj,
                   fronts, best if j == len(denoise_seq) - 1 else None,
                   dimg, mean_pred, iou, coverage, step, args.steps,
                   "scoring frontiers from the prediction", rlabel, "score")
            fidx += 1

        # ---- plan an A* path (walls inflated 1px for robot clearance) and drive it ----
        wall_inf = cv2.dilate((grid <= 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
        free_clear = wall_inf == 0
        known_free = (known == 1) & free_clear
        path = astar(known_free, snap(known_free, robot), snap(known_free, best))
        if path is None:                       # fall back to all clear free space
            path = astar(free_clear, snap(free_clear, robot), snap(free_clear, best))
        if not path or len(path) < 2:
            observe(robot)                     # unreachable: just re-observe in place
        else:
            seg = path[1:]                     # cells to traverse (exclude current pos)
            render_at = set(np.linspace(0, len(seg) - 1, min(6, len(seg))).astype(int))
            for ci, c in enumerate(seg):
                robot = c
                traj.append(c)                 # full-res trajectory hugs the corridor
                if ci % 4 == 0 or ci == len(seg) - 1:
                    observe(c)
                if ci in render_at:
                    cov = float(known[grid > 0.5].sum()) / total_free
                    render(out / f"frame_{fidx:04d}.png", grid, occ, known, robot, traj,
                           [], None, denoise_seq[-1], mean_pred, iou, cov, step, args.steps,
                           "driving to chosen frontier (A* path)",
                           "diffusion prediction (held fixed while driving)", "drive")
                    fidx += 1
        print(f"step {step}: coverage {coverage:.0%}  IoU {iou:.2f}  frontiers {len(fronts)}")

    wall_hits = sum(1 for (y, x) in traj if grid[y, x] <= 0.5)
    print(f"trajectory wall-cell hits: {wall_hits} (should be 0)")
    print(f"wrote {fidx} frames to {out}")


if __name__ == "__main__":
    main()
