"""Unit tests for the diffusion model components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from unet import ConditionalUNet, count_parameters
from diffusion import DDPMScheduler


def test_unet_forward():
    model = ConditionalUNet(in_channels=3, out_channels=1, base_channels=32,
                            channel_mults=(1, 2, 4, 4), time_dim=128)
    x = torch.randn(2, 1, 256, 256)
    t = torch.randint(0, 1000, (2,))
    partial = torch.randn(2, 1, 256, 256)
    mask = torch.ones(2, 1, 256, 256)

    out = model(x, t, partial, mask)
    assert out.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {out.shape}"


def test_unet_param_count():
    model = ConditionalUNet()
    params = count_parameters(model)
    assert 3_000_000 < params < 10_000_000, f"Expected ~4-5M params, got {params:,}"


def test_ddpm_q_sample():
    sched = DDPMScheduler(num_timesteps=1000, device="cpu")
    x0 = torch.randn(4, 1, 64, 64)
    t = torch.tensor([0, 250, 500, 999])

    x_noisy, noise = sched.q_sample(x0, t)
    assert x_noisy.shape == x0.shape
    assert noise.shape == x0.shape

    diff_t0 = (x_noisy[0] - x0[0]).abs().mean().item()
    diff_t999 = (x_noisy[3] - x0[3]).abs().mean().item()
    assert diff_t999 > diff_t0, "More noise at higher t"


def test_ddpm_p_sample():
    model = ConditionalUNet(base_channels=16, channel_mults=(1, 2))
    sched = DDPMScheduler(num_timesteps=100, device="cpu")
    x = torch.randn(1, 1, 64, 64)
    t = torch.tensor([50])
    partial = torch.randn(1, 1, 64, 64)
    mask = torch.ones(1, 1, 64, 64)

    out = sched.p_sample(model, x, t, partial, mask)
    assert out.shape == x.shape


if __name__ == "__main__":
    test_unet_forward()
    print("PASS: test_unet_forward")
    test_unet_param_count()
    print("PASS: test_unet_param_count")
    test_ddpm_q_sample()
    print("PASS: test_ddpm_q_sample")
    test_ddpm_p_sample()
    print("PASS: test_ddpm_p_sample")
    print("\nAll tests passed.")
