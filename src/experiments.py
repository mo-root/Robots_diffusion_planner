"""Rigorous experiments: does diffusion-guided exploration actually help, and when?

Exp 1: 30 maps, both methods, area-over-time curves with confidence intervals
Exp 3: lidar range sweep (30/50/70/100 px) on 10 maps to test prediction value
"""

import os
import sys
import math
import time
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from diffusion import DDPMScheduler
from unet import ConditionalUNet
from simulate_exploration import detect_frontier_clusters, iou


@torch.no_grad()
def score_diff(model, scheduler, partial, known_mask, clusters, robot_pos, device,
               K=8, info_radius=70, lam=0.5, beta=0.5):
    pm = torch.tensor(partial * 2 - 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    completions = []
    for _ in range(K):
        pred = scheduler.sample_ddim(model, pm, km, num_steps=30)
        completions.append((pred[0, 0].cpu().numpy() + 1) / 2)
    h, w = known_mask.shape
    ry, rx = robot_pos
    scored = []
    for cy, cx, sz in clusters:
        gains = []
        for comp in completions:
            r = info_radius
            y1, y2 = max(0, cy-r), min(h, cy+r+1)
            x1, x2 = max(0, cx-r), min(w, cx+r+1)
            gain = float(((comp[y1:y2, x1:x2] > 0.5) & (known_mask[y1:y2, x1:x2] < 0.5)).sum())
            gains.append(gain)
        exp = float(np.mean(gains))
        var = float(np.std(gains))
        dist = math.sqrt((cy-ry)**2 + (cx-rx)**2)
        scored.append((exp + lam*var - beta*dist, cy, cx, exp, var))
    scored.sort(reverse=True)
    return scored


def score_base(known_mask, clusters, robot_pos, beta=0.5, info_radius=70):
    h, w = known_mask.shape
    ry, rx = robot_pos
    scored = []
    for cy, cx, sz in clusters:
        r = info_radius
        y1, y2 = max(0, cy-r), min(h, cy+r+1)
        x1, x2 = max(0, cx-r), min(w, cx+r+1)
        unk = float((known_mask[y1:y2, x1:x2] < 0.5).sum())
        dist = math.sqrt((cy-ry)**2 + (cx-rx)**2)
        scored.append((unk - beta*dist, cy, cx, unk, 0))
    scored.sort(reverse=True)
    return scored


def run_exploration(grid, model, scheduler, device, mode='diffusion',
                    max_steps=20, target=0.95, lidar_range=70, seed=42):
    h, w = grid.shape
    rng = np.random.default_rng(seed)
    robot_pos = find_free_position(grid, rng)
    combined = np.zeros((h, w), dtype=np.uint8)
    coverages = []

    for step in range(max_steps):
        vis = simulate_lidar(grid, robot_pos, num_rays=360, max_range_px=lidar_range)
        combined = np.maximum(combined, vis)
        partial = np.full_like(grid, 0.5)
        partial[combined > 0] = grid[combined > 0]
        known = (combined > 0).astype(np.float32)

        free = (grid > 0.5).sum()
        known_free = ((grid > 0.5) & (combined > 0)).sum()
        cov = float(known_free / max(free, 1))
        coverages.append(cov)

        if cov >= target:
            break

        clusters = detect_frontier_clusters(known)
        if not clusters:
            break

        if mode == 'diffusion':
            scored = score_diff(model, scheduler, partial, known, clusters, robot_pos,
                                device, info_radius=lidar_range)
        else:
            scored = score_base(known, clusters, robot_pos, info_radius=lidar_range)

        best = scored[0]
        ry, rx = robot_pos
        ty, tx = best[1], best[2]
        dist = math.sqrt((ty-ry)**2 + (tx-rx)**2)
        n_int = max(1, int(dist / 20))
        for si in range(1, n_int + 1):
            frac = si / n_int
            iy = int(ry + (ty-ry) * frac)
            ix = int(rx + (tx-rx) * frac)
            if 0 <= iy < h and 0 <= ix < w and grid[iy, ix] > 0.5:
                iv = simulate_lidar(grid, (iy, ix), num_rays=360, max_range_px=lidar_range)
                combined = np.maximum(combined, iv)
        robot_pos = (best[1], best[2])

    return coverages


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = ConditionalUNet(
        in_channels=3, out_channels=1,
        base_channels=args.get('base_channels', 32),
        channel_mults=tuple(args.get('channel_mults', [1, 2, 4, 4])),
        time_dim=args.get('time_dim', 128),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    scheduler = DDPMScheduler(num_timesteps=args.get('T', 1000), device=device)
    return model, scheduler


def select_diverse_maps(json_files, n=30, seed=0):
    """Pick maps stratified by file index (rough proxy for diversity)."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(json_files), size=min(n * 2, len(json_files)), replace=False)
    return sorted(indices.tolist())[:n]


def room_count(grid):
    """Rough complexity proxy: count connected free-space regions after morphological closing."""
    from scipy import ndimage
    closed = ndimage.binary_opening(grid > 0.5, iterations=2)
    labeled, n = ndimage.label(closed)
    sizes = ndimage.sum(closed, labeled, range(1, n + 1))
    return int(np.sum(sizes > 200))


def run_experiment_1(model, scheduler, device, json_files, n_maps, out_dir, max_steps=20):
    """N maps, both methods, lidar=70. Save per-step coverage for both."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: N={n_maps} maps, diffusion vs baseline (lidar=70)")
    print(f"{'='*70}\n")

    test_indices = select_diverse_maps(json_files, n=n_maps)
    results = []
    csv_path = os.path.join(out_dir, "exp1_per_map.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["map_idx", "method", "step", "coverage"])

        for i, mi in enumerate(test_indices):
            try:
                plan = load_floor_plan(str(json_files[mi]))
                grid = rasterize_floor_plan(plan, 256)
            except Exception as e:
                print(f"  Map {mi}: load failed ({e}), skip")
                continue

            try:
                rooms = room_count(grid)
            except Exception:
                rooms = -1

            t0 = time.time()
            diff_cov = run_exploration(grid, model, scheduler, device, 'diffusion',
                                       max_steps=max_steps, lidar_range=70)
            dt = time.time() - t0

            t0 = time.time()
            base_cov = run_exploration(grid, model, scheduler, device, 'baseline',
                                       max_steps=max_steps, lidar_range=70)
            bt = time.time() - t0

            for step, c in enumerate(diff_cov):
                wr.writerow([mi, "diffusion", step + 1, c])
            for step, c in enumerate(base_cov):
                wr.writerow([mi, "baseline", step + 1, c])
            fh.flush()

            d80 = next((s + 1 for s, c in enumerate(diff_cov) if c >= 0.8), -1)
            b80 = next((s + 1 for s, c in enumerate(base_cov) if c >= 0.8), -1)
            d90 = next((s + 1 for s, c in enumerate(diff_cov) if c >= 0.9), -1)
            b90 = next((s + 1 for s, c in enumerate(base_cov) if c >= 0.9), -1)

            print(f"  [{i+1}/{len(test_indices)}] Map {mi:5d} rooms~{rooms}: "
                  f"D={diff_cov[-1]:.0%} ({dt:.0f}s) B={base_cov[-1]:.0%} ({bt:.0f}s) "
                  f"| 80%: D@{d80} B@{b80} | 90%: D@{d90} B@{b90}")

            results.append({
                "map_idx": mi, "rooms": rooms,
                "diff_final": diff_cov[-1], "base_final": base_cov[-1],
                "diff_steps": len(diff_cov), "base_steps": len(base_cov),
                "diff_to_80": d80, "base_to_80": b80,
                "diff_to_90": d90, "base_to_90": b90,
                "diff_cov": diff_cov, "base_cov": base_cov,
            })

    with open(os.path.join(out_dir, "exp1_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n  Wrote {csv_path}")
    print(f"  Wrote {out_dir}/exp1_results.json")
    return results


def run_experiment_3(model, scheduler, device, json_files, n_maps, out_dir, max_steps=20):
    """Lidar range sweep on N maps."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 3: lidar range sweep, N={n_maps} maps")
    print(f"{'='*70}\n")

    test_indices = select_diverse_maps(json_files, n=n_maps, seed=1)
    ranges = [30, 50, 70, 100]
    results = []
    csv_path = os.path.join(out_dir, "exp3_per_map.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["map_idx", "lidar", "method", "step", "coverage"])

        for i, mi in enumerate(test_indices):
            try:
                plan = load_floor_plan(str(json_files[mi]))
                grid = rasterize_floor_plan(plan, 256)
            except Exception:
                continue

            for lr in ranges:
                t0 = time.time()
                diff_cov = run_exploration(grid, model, scheduler, device, 'diffusion',
                                           max_steps=max_steps, lidar_range=lr)
                dt = time.time() - t0

                base_cov = run_exploration(grid, model, scheduler, device, 'baseline',
                                           max_steps=max_steps, lidar_range=lr)

                for step, c in enumerate(diff_cov):
                    wr.writerow([mi, lr, "diffusion", step + 1, c])
                for step, c in enumerate(base_cov):
                    wr.writerow([mi, lr, "baseline", step + 1, c])
                fh.flush()

                d80 = next((s + 1 for s, c in enumerate(diff_cov) if c >= 0.8), -1)
                b80 = next((s + 1 for s, c in enumerate(base_cov) if c >= 0.8), -1)

                print(f"  [{i+1}/{len(test_indices)}] Map {mi} lidar={lr}: "
                      f"D={diff_cov[-1]:.0%} B={base_cov[-1]:.0%} | 80%: D@{d80} B@{b80} ({dt:.0f}s)")

                results.append({
                    "map_idx": mi, "lidar": lr,
                    "diff_final": diff_cov[-1], "base_final": base_cov[-1],
                    "diff_to_80": d80, "base_to_80": b80,
                    "diff_cov": diff_cov, "base_cov": base_cov,
                })

    with open(os.path.join(out_dir, "exp3_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Wrote {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/checkpoints/model_epoch0020.pt")
    parser.add_argument("--json_dir",
                        default="/Users/moin/Robotics-class/final-project/data/HouseExpo/HouseExpo/json")
    parser.add_argument("--out_dir", default="results/experiments")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--exp1_maps", type=int, default=30)
    parser.add_argument("--exp3_maps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--skip_exp1", action="store_true")
    parser.add_argument("--skip_exp3", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    model, scheduler = load_model(args.checkpoint, device)

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    print(f"Found {len(json_files)} maps")

    t_total = time.time()

    if not args.skip_exp1:
        run_experiment_1(model, scheduler, device, json_files,
                         args.exp1_maps, args.out_dir, args.max_steps)

    if not args.skip_exp3:
        run_experiment_3(model, scheduler, device, json_files,
                         args.exp3_maps, args.out_dir, args.max_steps)

    print(f"\nTotal time: {(time.time() - t_total)/60:.1f} min")
