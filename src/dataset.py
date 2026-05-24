"""PyTorch Dataset for loading generated (partial_map, full_map) pairs."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MapCompletionDataset(Dataset):
    def __init__(self, data_dir: str):
        cache_path = os.path.join(data_dir, "_filelist.txt")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                names = [line.strip() for line in f if line.strip()]
        else:
            names = [n for n in os.listdir(data_dir) if n.endswith(".npz")]
            with open(cache_path, "w") as f:
                f.writelines(n + "\n" for n in names)
        self.files = [os.path.join(data_dir, n) for n in names]
        if not self.files:
            raise ValueError(f"No .npz files in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])["data"].astype(np.float32)

        partial_map = data[:, :, 0]
        known_mask = data[:, :, 1]
        full_map = data[:, :, 2]

        full_map = full_map * 2.0 - 1.0
        partial_map = partial_map * 2.0 - 1.0

        return {
            "partial_map": torch.tensor(partial_map).unsqueeze(0),
            "known_mask": torch.tensor(known_mask).unsqueeze(0),
            "full_map": torch.tensor(full_map).unsqueeze(0),
        }
