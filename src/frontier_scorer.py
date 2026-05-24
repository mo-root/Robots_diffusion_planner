"""Frontier scoring using diffusion-based map completion.

Given a partial occupancy grid, generates K map completions and scores
each frontier cell by expected information gain + uncertainty bonus.

score(f) = E[info_gain(f)] + lambda * Std[info_gain(f)] - beta * distance(f)

This module is standalone (no ROS). The ROS node wraps it.

Usage:
    scorer = FrontierScorer("results/checkpoints/model_final.pt", device="cuda")
    frontiers, scores = scorer.score_frontiers(partial_map, known_mask, robot_pos)
"""

import numpy as np
import torch

from diffusion import DDPMScheduler
from unet import ConditionalUNet


class FrontierScorer:
    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 K: int = 8, ddim_steps: int = 50,
                 lambda_var: float = 1.0, beta_dist: float = 0.5,
                 info_radius: int = 15):
        self.device = torch.device(device)
        self.K = K
        self.ddim_steps = ddim_steps
        self.lambda_var = lambda_var
        self.beta_dist = beta_dist
        self.info_radius = info_radius

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_args = ckpt.get("args", {})

        self.model = ConditionalUNet(
            in_channels=3, out_channels=1,
            base_channels=ckpt_args.get("base_channels", 32),
            channel_mults=tuple(ckpt_args.get("channel_mults", [1, 2, 4, 4])),
            time_dim=ckpt_args.get("time_dim", 128),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.scheduler = DDPMScheduler(
            num_timesteps=ckpt_args.get("T", 1000), device=self.device
        )

    def detect_frontiers(self, known_mask: np.ndarray, min_unknown_neighbors: int = 3):
        """Find frontier cells: known cells adjacent to unknown cells."""
        h, w = known_mask.shape
        frontiers = []

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if known_mask[y, x] < 0.5:
                    continue
                unknown_count = 0
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx] < 0.5:
                        unknown_count += 1
                if unknown_count >= min_unknown_neighbors:
                    frontiers.append((y, x))

        return frontiers

    def compute_info_gain(self, predicted_map: np.ndarray, known_mask: np.ndarray,
                          frontier_pos: tuple) -> float:
        """Count predicted free cells near a frontier that are currently unknown."""
        fy, fx = frontier_pos
        r = self.info_radius
        h, w = predicted_map.shape

        y_min = max(0, fy - r)
        y_max = min(h, fy + r + 1)
        x_min = max(0, fx - r)
        x_max = min(w, fx + r + 1)

        region_pred = predicted_map[y_min:y_max, x_min:x_max]
        region_mask = known_mask[y_min:y_max, x_min:x_max]

        new_free = ((region_pred > 0.5) & (region_mask < 0.5)).sum()
        return float(new_free)

    @torch.no_grad()
    def generate_completions(self, partial_map: np.ndarray,
                             known_mask: np.ndarray) -> list:
        """Generate K map completions from the partial map."""
        pm = torch.tensor(partial_map * 2.0 - 1.0, dtype=torch.float32)
        pm = pm.unsqueeze(0).unsqueeze(0).to(self.device)
        km = torch.tensor(known_mask, dtype=torch.float32)
        km = km.unsqueeze(0).unsqueeze(0).to(self.device)

        completions = []
        for _ in range(self.K):
            pred = self.scheduler.sample_ddim(
                self.model, pm, km, num_steps=self.ddim_steps
            )
            pred_np = (pred[0, 0].cpu().numpy() + 1) / 2
            completions.append(pred_np.clip(0, 1))

        return completions

    def score_frontiers(self, partial_map: np.ndarray, known_mask: np.ndarray,
                        robot_pos: tuple = None) -> tuple:
        """Score all frontier cells. Returns (frontiers, scores, completions)."""
        frontiers = self.detect_frontiers(known_mask)
        if not frontiers:
            return [], [], []

        completions = self.generate_completions(partial_map, known_mask)

        scores = []
        for fy, fx in frontiers:
            gains = [self.compute_info_gain(c, known_mask, (fy, fx)) for c in completions]
            expected_gain = np.mean(gains)
            gain_std = np.std(gains)

            dist = 0.0
            if robot_pos is not None:
                ry, rx = robot_pos
                dist = np.sqrt((fy - ry)**2 + (fx - rx)**2)

            score = expected_gain + self.lambda_var * gain_std - self.beta_dist * dist
            scores.append(score)

        sorted_idx = np.argsort(scores)[::-1]
        frontiers = [frontiers[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]

        return frontiers, scores, completions

    def get_best_frontier(self, partial_map: np.ndarray, known_mask: np.ndarray,
                          robot_pos: tuple = None) -> tuple:
        """Return the single best frontier to explore."""
        frontiers, scores, completions = self.score_frontiers(
            partial_map, known_mask, robot_pos
        )
        if not frontiers:
            return None, None, completions
        return frontiers[0], scores[0], completions
