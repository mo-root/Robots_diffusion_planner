"""
HouseExpo -> (partial_map, full_map) training pair generator.

Pipeline:
1. Load JSON polygon floor plan from HouseExpo
2. Rasterize to 256x256 binary occupancy grid (0=free, 1=wall)
3. Pick random robot position (must be in free space)
4. Simulate lidar raycasting from that position
5. Build partial map: visible cells keep their value, rest = 0.5 (unknown)
6. Save (partial_map, full_map) as .npz
7. Augment with rotations and flips

Usage:
    python src/data_generator.py --json_dir data/HouseExpo/HouseExpo/json \
                                  --out_dir data/train \
                                  --samples_per_map 10 \
                                  --resolution 256 \
                                  --num_workers 8
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np


def load_floor_plan(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def rasterize_floor_plan(plan: dict, resolution: int = 256) -> np.ndarray:
    """Convert polygon vertices to a binary occupancy grid.

    Returns grid where 1.0 = free space, 0.0 = wall/outside.
    """
    verts = np.array(plan["verts"], dtype=np.float64)
    bbox = plan["bbox"]

    x_min, y_min = bbox["min"]
    x_max, y_max = bbox["max"]

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    max_dim = max(width, height)
    padding = max_dim * 0.05

    scale = (resolution - 2) / (max_dim + 2 * padding)

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    pixel_verts = np.zeros_like(verts, dtype=np.int32)
    pixel_verts[:, 0] = ((verts[:, 0] - cx) * scale + resolution / 2).astype(np.int32)
    pixel_verts[:, 1] = ((verts[:, 1] - cy) * scale + resolution / 2).astype(np.int32)

    grid = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(grid, [pixel_verts], 1)

    return grid.astype(np.float32)


def find_free_position(grid: np.ndarray, rng: np.random.Generator) -> tuple:
    """Pick a random free cell (value == 1.0) that's not too close to walls."""
    kernel = np.ones((5, 5), dtype=np.uint8)
    eroded = cv2.erode((grid > 0.5).astype(np.uint8), kernel)
    free_cells = np.argwhere(eroded > 0)

    if len(free_cells) < 10:
        free_cells = np.argwhere(grid > 0.5)

    if len(free_cells) == 0:
        return None

    idx = rng.integers(0, len(free_cells))
    return int(free_cells[idx, 0]), int(free_cells[idx, 1])


def simulate_lidar(grid: np.ndarray, robot_pos: tuple,
                   num_rays: int = 360, max_range_px: int = 80) -> np.ndarray:
    """Simulate 2D lidar from robot_pos. Returns visibility mask (1=visible, 0=not)."""
    h, w = grid.shape
    ry, rx = robot_pos

    visible = np.zeros((h, w), dtype=np.uint8)
    visible[ry, rx] = 1

    for i in range(num_rays):
        angle = 2.0 * math.pi * i / num_rays
        dx = math.cos(angle)
        dy = math.sin(angle)

        for step in range(1, max_range_px + 1):
            x = rx + dx * step
            y = ry + dy * step

            xi, yi = int(round(x)), int(round(y))

            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                break

            visible[yi, xi] = 1

            if grid[yi, xi] < 0.5:
                break

    return visible


def make_partial_map(grid: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Create partial map: visible cells keep value, rest = 0.5 (unknown).

    Channel layout (H, W, 3):
      ch0: occupancy (0=wall, 1=free, 0.5=unknown)
      ch1: known mask (1=observed, 0=unknown)
      ch2: full map (ground truth, only used as target)
    """
    partial = np.full_like(grid, 0.5)
    partial[visible > 0] = grid[visible > 0]

    known_mask = (visible > 0).astype(np.float32)

    return np.stack([partial, known_mask, grid], axis=-1)


def augment_pair(partial_3ch: np.ndarray) -> list:
    """Generate rotated/flipped variants. Returns list of augmented arrays."""
    results = [partial_3ch]

    for k in [1, 2, 3]:
        results.append(np.rot90(partial_3ch, k=k, axes=(0, 1)).copy())

    flipped = np.flip(partial_3ch, axis=1).copy()
    results.append(flipped)
    for k in [1, 2, 3]:
        results.append(np.rot90(flipped, k=k, axes=(0, 1)).copy())

    return results


def process_single_map(args):
    """Process one floor plan JSON file. Returns number of samples generated."""
    json_path, out_dir, samples_per_map, resolution, do_augment, seed = args

    rng = np.random.default_rng(seed)

    try:
        plan = load_floor_plan(json_path)
    except Exception:
        return 0

    grid = rasterize_floor_plan(plan, resolution)
    if grid is None:
        return 0

    free_ratio = grid.sum() / grid.size
    if free_ratio < 0.05 or free_ratio > 0.95:
        return 0

    map_id = Path(json_path).stem
    count = 0

    for s in range(samples_per_map):
        pos = find_free_position(grid, rng)
        if pos is None:
            continue

        max_range = rng.integers(40, 100)
        visible = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_range)

        vis_ratio = visible.sum() / (grid > 0.5).sum()
        if vis_ratio < 0.05 or vis_ratio > 0.95:
            continue

        sample = make_partial_map(grid, visible)

        if do_augment:
            augmented = augment_pair(sample)
        else:
            augmented = [sample]

        for ai, aug_sample in enumerate(augmented):
            fname = f"{map_id}_s{s}_a{ai}.npz"
            np.savez_compressed(
                os.path.join(out_dir, fname),
                data=aug_sample.astype(np.float16)
            )
            count += 1

    return count


def generate_dataset(json_dir: str, out_dir: str, samples_per_map: int = 10,
                     resolution: int = 256, do_augment: bool = True,
                     num_workers: int = 8, max_maps: int = None):
    os.makedirs(out_dir, exist_ok=True)

    json_files = sorted(Path(json_dir).glob("*.json"))
    if max_maps:
        json_files = json_files[:max_maps]

    print(f"Found {len(json_files)} floor plans")
    print(f"Generating {samples_per_map} samples/map, augment={do_augment}")
    print(f"Output: {out_dir}")

    tasks = [
        (str(f), out_dir, samples_per_map, resolution, do_augment, i * 1000 + 42)
        for i, f in enumerate(json_files)
    ]

    total = 0
    done = 0

    with Pool(num_workers) as pool:
        for count in pool.imap_unordered(process_single_map, tasks, chunksize=50):
            total += count
            done += 1
            if done % 500 == 0:
                print(f"  [{done}/{len(json_files)}] maps processed, {total} samples so far")

    print(f"\nDone! Generated {total} training samples from {len(json_files)} floor plans.")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate diffusion training data from HouseExpo")
    parser.add_argument("--json_dir", type=str,
                        default="data/HouseExpo/HouseExpo/json")
    parser.add_argument("--out_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--samples_per_map", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--max_maps", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    json_files = sorted(Path(args.json_dir).glob("*.json"))
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(json_files))

    n_val = int(len(json_files) * args.val_split)
    val_indices = set(indices[:n_val])
    train_indices = set(indices[n_val:])

    train_files = [json_files[i] for i in sorted(train_indices)]
    val_files = [json_files[i] for i in sorted(val_indices)]

    if args.max_maps:
        train_files = train_files[:args.max_maps]
        val_files = val_files[:max(10, args.max_maps // 20)]

    print(f"=== Train set: {len(train_files)} maps ===")
    os.makedirs(args.out_dir, exist_ok=True)
    train_tasks = [
        (str(f), args.out_dir, args.samples_per_map, args.resolution,
         not args.no_augment, i * 1000 + 42)
        for i, f in enumerate(train_files)
    ]

    total_train = 0
    done = 0
    with Pool(args.num_workers) as pool:
        for count in pool.imap_unordered(process_single_map, train_tasks, chunksize=50):
            total_train += count
            done += 1
            if done % 500 == 0:
                print(f"  [{done}/{len(train_files)}] maps, {total_train} samples")

    print(f"Train: {total_train} samples from {len(train_files)} maps\n")

    print(f"=== Val set: {len(val_files)} maps ===")
    os.makedirs(args.val_dir, exist_ok=True)
    val_tasks = [
        (str(f), args.val_dir, args.samples_per_map, args.resolution,
         False, i * 2000 + 99)
        for i, f in enumerate(val_files)
    ]

    total_val = 0
    done = 0
    with Pool(args.num_workers) as pool:
        for count in pool.imap_unordered(process_single_map, val_tasks, chunksize=50):
            total_val += count
            done += 1

    print(f"Val: {total_val} samples from {len(val_files)} maps")
    print(f"\nTotal: {total_train + total_val} samples ({total_train} train + {total_val} val)")
