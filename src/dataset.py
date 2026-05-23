"""PyTorch Dataset for loading generated (partial_map, full_map) pairs."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MapCompletionDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(Path(data_dir).glob("*.npz"))
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
