"""Showcase artifacts using multi-scan (high coverage) input.

Same visualizations as the original showcase, but with 3-5 robot positions
as input -- showing how much better predictions are with realistic coverage.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import load_floor_plan, rasterize_floor_plan, find_free_position, simulate_lidar
from diffusion import DDPMScheduler
from unet import ConditionalUNet

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL required")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ConditionalUNet(
        in_channels=3, out_channels=1,
        base_channels=args.get("base_channels", 32),
        channel_mults=tuple(args.get("channel_mults", [1, 2, 4, 4])),
        time_dim=args.get("time_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scheduler = DDPMScheduler(num_timesteps=args.get("T", 1000), device=device)
    return model, scheduler


def partial_rgb(p):
    vis = np.zeros((*p.shape, 3))
    vis[p > 0.6] = [1, 1, 1]
    vis[p < 0.4] = [0, 0, 0]
    vis[(p >= 0.4) & (p <= 0.6)] = [0.5, 0.5, 0.7]
    return vis


def iou(gt, pred):
    g, p = gt > 0.5, pred > 0.5
    return float((g & p).sum()) / max(float((g | p).sum()), 1)


def fig_to_frame(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    return Image.fromarray(np.asarray(buf).reshape(h, w, 4)[:, :, :3].copy())


def make_multi_scan(grid, n_positions, rng, max_range=70):
    h, w = grid.shape
    combined = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n_positions):
        pos = find_free_position(grid, rng)
        if pos is None:
            continue
        vis = simulate_lidar(grid, pos, num_rays=360, max_range_px=max_range)
        combined = np.maximum(combined, vis)
    partial = np.full_like(grid, 0.5)
    partial[combined > 0] = grid[combined > 0]
    known = (combined > 0).astype(np.float32)
    return partial, known


@torch.no_grad()
def predict(model, scheduler, partial, known, device, steps=50):
    pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    km = torch.tensor(known, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    pred = scheduler.sample_ddim(model, pm, km, num_steps=steps)
    return (pred[0, 0].cpu().numpy() + 1) / 2, pm, km


@torch.no_grad()
def run_ddim_intermediates(model, scheduler, pm, km, device, steps=50):
    x = torch.randn(1, 1, 256, 256, device=device)
    step_size = scheduler.T // steps
    timesteps = list(range(0, scheduler.T, step_size))[::-1]
    results = []
    for i, t_val in enumerate(timesteps):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        pred_noise = model(x, t, pm, km)
        ac = scheduler.alphas_cumprod[t_val]
        ac_prev = scheduler.alphas_cumprod[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0, device=device)
        pred_x0 = ((x - torch.sqrt(1 - ac) * pred_noise) / torch.sqrt(ac)).clamp(-1, 1)
        x = torch.sqrt(ac_prev) * pred_x0 + torch.sqrt(1 - ac_prev) * pred_noise
        results.append((t_val, x.clone()))
    return results


def hc_1_comparison_grid(model, scheduler, json_dir, device, save_path, n=10, n_scans=5):
    """Comparison grid with multi-scan input."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(json_files), size=n, replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(16, 3.5 * n))
    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        partial, known = make_multi_scan(grid, n_scans, np.random.default_rng(row * 10))
        predicted, _, _ = predict(model, scheduler, partial, known, device)
        predicted = predicted.clip(0, 1)
        score = iou(grid, predicted)

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("Ground Truth" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_rgb(partial))
        axes[row, 1].set_title(f"Partial ({known.mean():.0%}, {n_scans} scans)" if row == 0 else f"({known.mean():.0%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(predicted, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted" if row == 0 else "")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(np.abs(grid - predicted), cmap="hot", vmin=0, vmax=1)
        color = "green" if score > 0.7 else "orange"
        axes[row, 3].set_title(f"Error (IoU: {score:.2f})" if row == 0 else f"IoU: {score:.2f}",
                                color=color, fontweight="bold")
        axes[row, 3].axis("off")

    plt.suptitle(f"High-Coverage Predictions ({n_scans} scans per map, ~30-40% visible)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[1/5] Comparison grid saved: {save_path}")


def hc_2_denoising_gifs(model, scheduler, json_dir, device, out_dir, n_gifs=6, n_scans=5):
    """Multiple denoising GIFs with high coverage input."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(55)
    indices = rng.choice(len(json_files), size=n_gifs, replace=False)

    for gi, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        partial, known = make_multi_scan(grid, n_scans, np.random.default_rng(gi * 7))
        pm = torch.tensor(partial * 2.0 - 1.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        km = torch.tensor(known, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        intermediates = run_ddim_intermediates(model, scheduler, pm, km, device)
        pick = intermediates[::max(1, len(intermediates)//20)]
        if intermediates[-1] not in pick:
            pick.append(intermediates[-1])

        frames = []
        for t_val, x in pick:
            current = (x[0, 0].cpu().numpy() + 1) / 2
            progress = (1 - t_val / scheduler.T) * 100
            score = iou(grid, current.clip(0, 1))

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            axes[0].imshow(partial_rgb(partial))
            axes[0].set_title(f"Input ({known.mean():.0%} known)", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(current.clip(0,1), cmap="gray_r", vmin=0, vmax=1)
            axes[1].set_title(f"Denoising: {progress:.0f}%", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
            axes[2].set_title("Ground Truth", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(np.abs(grid - current.clip(0,1)), cmap="hot", vmin=0, vmax=1)
            axes[3].set_title(f"Error (IoU: {score:.2f})", fontsize=12)
            axes[3].axis("off")

            plt.suptitle(f"High-Coverage Denoising ({n_scans} scans)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            frames.append(fig_to_frame(fig))
            plt.close()

        for _ in range(8):
            frames.append(frames[-1])

        path = os.path.join(out_dir, f"hc_denoise_{gi+1:02d}.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print(f"  GIF {gi+1}/{n_gifs}: {path} ({len(frames)} frames)")

    print(f"[2/5] All denoising GIFs saved")


def hc_3_diversity(model, scheduler, json_dir, device, save_path, n_maps=4, K=8, n_scans=5):
    """K=8 completions from high coverage input."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(66)
    indices = rng.choice(len(json_files), size=n_maps, replace=False)

    fig, axes = plt.subplots(n_maps, K + 2, figsize=(3 * (K + 2), 3.5 * n_maps))

    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        partial, known = make_multi_scan(grid, n_scans, np.random.default_rng(row * 13))

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("GT" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_rgb(partial))
        axes[row, 1].set_title(f"({known.mean():.0%})" if row == 0 else f"({known.mean():.0%})")
        axes[row, 1].axis("off")

        for k in range(K):
            pred, _, _ = predict(model, scheduler, partial, known, device)
            pred = pred.clip(0, 1)
            score = iou(grid, pred)
            axes[row, k+2].imshow(pred, cmap="gray_r", vmin=0, vmax=1)
            axes[row, k+2].set_title(f"S{k+1} ({score:.2f})" if row == 0 else f"{score:.2f}")
            axes[row, k+2].axis("off")

    plt.suptitle(f"High-Coverage Diversity: {K} completions per map ({n_scans} scans input)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[3/5] Diversity grid saved: {save_path}")


def hc_4_uncertainty(model, scheduler, json_dir, device, save_path, n_maps=4, K=8, n_scans=5):
    """Uncertainty maps with high coverage."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(77)
    indices = rng.choice(len(json_files), size=n_maps, replace=False)

    fig, axes = plt.subplots(n_maps, 4, figsize=(16, 4 * n_maps))

    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        partial, known = make_multi_scan(grid, n_scans, np.random.default_rng(row * 17))

        preds = []
        for _ in range(K):
            p, _, _ = predict(model, scheduler, partial, known, device)
            preds.append(p.clip(0, 1))

        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0)
        score = iou(grid, mean_pred)

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title("GT" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(partial_rgb(partial))
        axes[row, 1].set_title(f"Partial ({known.mean():.0%})" if row == 0 else f"({known.mean():.0%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mean_pred, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Mean Pred (IoU: {score:.2f})" if row == 0 else f"IoU: {score:.2f}")
        axes[row, 2].axis("off")

        im = axes[row, 3].imshow(std_pred, cmap="hot", vmin=0, vmax=max(0.01, std_pred.max()))
        axes[row, 3].set_title(f"Uncertainty (avg: {std_pred.mean():.3f})" if row == 0 else f"avg: {std_pred.mean():.3f}")
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

    plt.suptitle(f"High-Coverage Uncertainty Maps ({n_scans} scans, K={K})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[4/5] Uncertainty maps saved: {save_path}")


def hc_5_side_by_side_low_vs_high(model, scheduler, json_dir, device, save_path, n_maps=6):
    """Direct comparison: 1 scan vs 5 scans prediction quality."""
    json_files = sorted(Path(json_dir).glob("*.json"))
    rng = np.random.default_rng(88)
    indices = rng.choice(len(json_files), size=n_maps, replace=False)

    fig, axes = plt.subplots(n_maps, 5, figsize=(20, 3.5 * n_maps))
    col_labels = ["Ground Truth", "1 scan (low)", "Predicted (low)", "5 scans (high)", "Predicted (high)"]

    for row, idx in enumerate(indices):
        plan = load_floor_plan(str(json_files[idx]))
        grid = rasterize_floor_plan(plan, 256)
        if grid is None:
            continue

        axes[row, 0].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 0].set_title(col_labels[0] if row == 0 else "")
        axes[row, 0].axis("off")

        p1, k1 = make_multi_scan(grid, 1, np.random.default_rng(row * 5))
        pred1, _, _ = predict(model, scheduler, p1, k1, device)
        pred1 = pred1.clip(0, 1)
        s1 = iou(grid, pred1)

        axes[row, 1].imshow(partial_rgb(p1))
        axes[row, 1].set_title(f"{col_labels[1]} ({k1.mean():.0%})" if row == 0 else f"({k1.mean():.0%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred1, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 2].set_title(f"IoU: {s1:.2f}" if row == 0 else f"{s1:.2f}",
                                color="orange", fontweight="bold")
        axes[row, 2].axis("off")

        p5, k5 = make_multi_scan(grid, 5, np.random.default_rng(row * 5 + 100))
        pred5, _, _ = predict(model, scheduler, p5, k5, device)
        pred5 = pred5.clip(0, 1)
        s5 = iou(grid, pred5)

        axes[row, 3].imshow(partial_rgb(p5))
        axes[row, 3].set_title(f"{col_labels[3]} ({k5.mean():.0%})" if row == 0 else f"({k5.mean():.0%})")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(pred5, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 4].set_title(f"IoU: {s5:.2f}" if row == 0 else f"{s5:.2f}",
                                color="green", fontweight="bold")
        axes[row, 4].axis("off")

    plt.suptitle("Low Coverage (1 scan) vs High Coverage (5 scans) — Same Model",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[5/5] Low vs high comparison saved: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/high_coverage_showcase")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print("Loading model...")
    model, scheduler = load_model(args.checkpoint, device)
    print()

    hc_1_comparison_grid(model, scheduler, args.json_dir, device,
                          os.path.join(args.out_dir, "01_comparison_grid.png"))
    hc_2_denoising_gifs(model, scheduler, args.json_dir, device, args.out_dir)
    hc_3_diversity(model, scheduler, args.json_dir, device,
                    os.path.join(args.out_dir, "03_diversity.png"))
    hc_4_uncertainty(model, scheduler, args.json_dir, device,
                      os.path.join(args.out_dir, "04_uncertainty.png"))
    hc_5_side_by_side_low_vs_high(model, scheduler, args.json_dir, device,
                                    os.path.join(args.out_dir, "05_low_vs_high.png"))

    print(f"\nAll high-coverage artifacts saved to {args.out_dir}/")
