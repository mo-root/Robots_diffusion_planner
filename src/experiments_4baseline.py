"""4-baseline ablation requested by Prof. Quattrini Li (2026-05-31 check-in).

Runs the same N=30 HouseExpo maps with four frontier scorers:
    1. nearest        — argmin distance to robot
    2. info_gain      — argmax E[unknown cells in sensor footprint] (no distance)
    3. heuristic      — unknown - beta * distance  (this is `baseline` from exp1)
    4. diffusion      — E[gain] + lam * Std[gain] - beta * distance over K=8 completions

Uses identical map selection (seed=0) and robot-placement seed (42) as exp1, so
the diffusion column is comparable to results/experiments/exp1_results.json.
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from diffusion import DDPMScheduler
from unet import ConditionalUNet
from simulate_exploration import detect_frontier_clusters
from experiments import score_diff, score_base, load_model, select_diverse_maps


def score_nearest(clusters, robot_pos):
    ry, rx = robot_pos
    scored = []
    for cy, cx, sz in clusters:
        dist = math.sqrt((cy - ry) ** 2 + (cx - rx) ** 2)
        scored.append((-dist, cy, cx, 0.0, 0.0))
    scored.sort(reverse=True)
    return scored


def score_info_gain(known_mask, clusters, robot_pos, info_radius=70):
    h, w = known_mask.shape
    scored = []
    for cy, cx, sz in clusters:
        r = info_radius
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        unk = float((known_mask[y1:y2, x1:x2] < 0.5).sum())
        scored.append((unk, cy, cx, unk, 0.0))
    scored.sort(reverse=True)
    return scored


def run_rollout(grid, model, scheduler, device, method,
                max_steps=20, target=0.95, lidar_range=70, seed=42):
    """Same loop as run_exploration in experiments.py, but with 4 method options."""
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
        clusters = [c for c in clusters if math.hypot(c[0] - robot_pos[0], c[1] - robot_pos[1]) >= 10]
        if not clusters:
            break

        if method == 'nearest':
            scored = score_nearest(clusters, robot_pos)
        elif method == 'info_gain':
            scored = score_info_gain(known, clusters, robot_pos, info_radius=lidar_range)
        elif method == 'heuristic':
            scored = score_base(known, clusters, robot_pos, info_radius=lidar_range)
        elif method == 'diffusion':
            scored = score_diff(model, scheduler, partial, known, clusters, robot_pos,
                                device, info_radius=lidar_range)
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
                iv = simulate_lidar(grid, (iy, ix), num_rays=360, max_range_px=lidar_range)
                combined = np.maximum(combined, iv)
        robot_pos = (best[1], best[2])

    return coverages


METHODS = ['nearest', 'info_gain', 'heuristic', 'diffusion']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='results/checkpoints/model_epoch0020.pt')
    p.add_argument('--json_dir', default='/Users/moin/Robotics-class/final-project/data/HouseExpo/HouseExpo/json')
    p.add_argument('--out_dir', default='results/experiments_4baseline')
    p.add_argument('--device', default='mps')
    p.add_argument('--n_maps', type=int, default=30)
    p.add_argument('--max_steps', type=int, default=20)
    p.add_argument('--skip_diffusion', action='store_true',
                   help='Skip diffusion rollouts and reuse exp1_results.json for that column')
    p.add_argument('--reuse_diffusion_from',
                   default='results/experiments/exp1_results.json',
                   help='Path to exp1_results.json whose diff_cov is reused as the diffusion column')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.skip_diffusion:
        model, scheduler = None, None
        with open(args.reuse_diffusion_from) as fh:
            diff_cache = {r['map_idx']: r['diff_cov'] for r in json.load(fh)}
        print(f'Reusing diffusion column from {args.reuse_diffusion_from} '
              f'({len(diff_cache)} maps)')
    else:
        print(f'Loading model from {args.checkpoint}...')
        model, scheduler = load_model(args.checkpoint, device)
        diff_cache = {}

    json_files = sorted(Path(args.json_dir).glob('*.json'))
    test_indices = select_diverse_maps(json_files, n=args.n_maps)
    print(f'N={len(test_indices)} maps. Methods: {METHODS}')

    rows = []
    t0 = time.time()
    for i, mi in enumerate(test_indices):
        try:
            plan = load_floor_plan(str(json_files[mi]))
            grid = rasterize_floor_plan(plan, 256)
        except Exception as e:
            print(f'  Map {mi}: load failed ({e}), skip')
            continue

        per_method = {}
        for m in METHODS:
            if m == 'diffusion' and args.skip_diffusion:
                if mi not in diff_cache:
                    print(f'  Map {mi}: missing in reuse cache, skip')
                    per_method[m] = None
                    continue
                per_method[m] = diff_cache[mi]
                continue
            cov = run_rollout(grid, model, scheduler, device, m,
                              max_steps=args.max_steps, lidar_range=70)
            per_method[m] = cov

        row = {'map_idx': int(mi)}
        for m, cov in per_method.items():
            row[f'{m}_cov'] = cov
            if cov is None:
                continue
            row[f'{m}_final'] = float(cov[-1])
            row[f'{m}_to_80'] = next((s + 1 for s, c in enumerate(cov) if c >= 0.8), -1)
            row[f'{m}_step4'] = float(cov[3]) if len(cov) > 3 else float(cov[-1])
        rows.append(row)

        line = ' | '.join(
            f"{m}={per_method[m][-1]:.0%}" if per_method[m] is not None else f"{m}=NA"
            for m in METHODS
        )
        elapsed = (time.time() - t0) / 60
        print(f'  [{i + 1}/{len(test_indices)}] Map {mi:5d} | {line} | {elapsed:.1f} min total')

        with open(os.path.join(args.out_dir, 'partial_results.json'), 'w') as fh:
            json.dump(rows, fh, indent=2)

    with open(os.path.join(args.out_dir, 'results.json'), 'w') as fh:
        json.dump(rows, fh, indent=2)

    print(f'\nWrote {args.out_dir}/results.json')
    print(f'Total time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
