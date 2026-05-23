"""Visualize generated training samples to verify data quality."""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(data_dir: str, num_samples: int = 10, save_path: str = None):
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return

    indices = np.random.default_rng(42).choice(len(files), size=min(num_samples, len(files)), replace=False)
    chosen = [files[i] for i in sorted(indices)]

    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for row, fpath in enumerate(chosen):
        data = np.load(fpath)["data"].astype(np.float32)
        partial = data[:, :, 0]
        known_mask = data[:, :, 1]
        full_map = data[:, :, 2]

        axes[row, 0].imshow(full_map, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("Full Map (GT)")
        axes[row, 0].axis("off")

        vis = np.zeros((*partial.shape, 3))
        vis[partial > 0.7] = [1, 1, 1]      # free = white
        vis[partial < 0.3] = [0, 0, 0]      # wall = black
        vis[(partial > 0.3) & (partial < 0.7)] = [0.5, 0.5, 0.7]  # unknown = blue-gray
        axes[row, 1].imshow(vis)
        axes[row, 1].set_title(f"Partial Map ({known_mask.mean():.0%} known)")
        axes[row, 1].axis("off")

        diff = np.zeros((*partial.shape, 3))
        diff[full_map > 0.5] = [0.9, 0.9, 0.9]
        diff[(full_map > 0.5) & (known_mask < 0.5)] = [0.3, 0.8, 0.3]  # unseen free = green
        diff[(full_map < 0.5) & (known_mask < 0.5)] = [0.8, 0.3, 0.3]  # unseen wall = red
        diff[(full_map < 0.5) & (known_mask > 0.5)] = [0, 0, 0]        # seen wall = black
        axes[row, 2].imshow(diff)
        axes[row, 2].set_title("Hidden regions (green=free, red=wall)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    visualize_samples(args.data_dir, args.num_samples, args.save)
