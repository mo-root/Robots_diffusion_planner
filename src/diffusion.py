"""DDPM diffusion scheduler -- forward process (add noise) and reverse process (denoise).

Based on Ho et al. NeurIPS 2020 "Denoising Diffusion Probabilistic Models".
"""

import torch
import torch.nn.functional as F
import numpy as np


class DDPMScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.T = num_timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> tuple:
        """Forward process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_noisy = sqrt_ac * x0 + sqrt_omac * noise
        return x_noisy, noise

    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor,
                 partial_map: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        """Single reverse step: denoise x_t to x_{t-1}."""
        with torch.no_grad():
            pred_noise = model(x_t, t, partial_map, known_mask)

            alpha = self.alphas[t][:, None, None, None]
            alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
            beta = self.betas[t][:, None, None, None]
            sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

            mean = (1.0 / torch.sqrt(alpha)) * (x_t - beta / sqrt_omac * pred_noise)

            if (t > 0).any():
                var = self.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(x_t)
                mask = (t > 0).float()[:, None, None, None]
                mean = mean + mask * torch.sqrt(var) * noise

            return mean

    @torch.no_grad()
    def sample(self, model, partial_map: torch.Tensor, known_mask: torch.Tensor,
               num_steps: int = None) -> torch.Tensor:
        """Full reverse process: generate a complete map from noise."""
        device = partial_map.device
        b = partial_map.shape[0]
        T = num_steps or self.T

        x = torch.randn(b, 1, partial_map.shape[2], partial_map.shape[3], device=device)

        for i in reversed(range(T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, partial_map, known_mask)

        return x

    @torch.no_grad()
    def sample_ddim(self, model, partial_map: torch.Tensor, known_mask: torch.Tensor,
                    num_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        """DDIM sampling for faster inference (Song et al. 2020)."""
        device = partial_map.device
        b = partial_map.shape[0]

        step_size = self.T // num_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = torch.randn(b, 1, partial_map.shape[2], partial_map.shape[3], device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((b,), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, t, partial_map, known_mask)

            alpha_cumprod_t = self.alphas_cumprod[t_val]
            alpha_cumprod_prev = (
                self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else
                torch.tensor(1.0, device=device)
            )

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            sigma = eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * pred_noise
            noise = torch.randn_like(x) if t_val > 0 else 0

            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise

        return x
