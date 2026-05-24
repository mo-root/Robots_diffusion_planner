"""Conditional U-Net for map completion diffusion model.

Input: concatenation of noisy target (1ch) + partial map (1ch) + known mask (1ch) + timestep embedding
Output: predicted noise (1ch), same spatial size as input

Architecture: standard encoder-decoder with skip connections, ~5M params.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        flat = normed.view(b, c, h * w).permute(0, 2, 1)
        out, _ = self.attn(flat, flat, flat)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + out


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, has_attn: bool = False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.res(x, t_emb)
        h = self.attn(h)
        return h, self.down(h)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, has_attn: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + out_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.res(x, t_emb)
        return self.attn(h)


class ConditionalUNet(nn.Module):
    """U-Net with 3 input channels: noisy_target (1) + partial_map (1) + known_mask (1)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_channels: int = 32, channel_mults=(1, 2, 4, 4),
                 time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]

        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            has_attn = i == len(channels) - 1
            self.downs.append(DownBlock(in_ch, out_ch, time_dim, has_attn))
            in_ch = out_ch

        self.mid_res1 = ResBlock(channels[-1], channels[-1], time_dim)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_res2 = ResBlock(channels[-1], channels[-1], time_dim)

        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            has_attn = i == 0
            up_in = channels[-1] if i == 0 else channels[len(channels) - i]
            self.ups.append(UpBlock(up_in, out_ch, time_dim, has_attn))

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                partial_map: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x = torch.cat([x_noisy, partial_map, known_mask], dim=1)
        x = self.input_conv(x)

        skips = []
        for down in self.downs:
            skip, x = down(x, t_emb)
            skips.append(skip)

        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ConditionalUNet()
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(2, 1, 256, 256)
    t = torch.randint(0, 1000, (2,))
    p = torch.randn(2, 1, 256, 256)
    m = torch.ones(2, 1, 256, 256)
    out = model(x, t, p, m)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
