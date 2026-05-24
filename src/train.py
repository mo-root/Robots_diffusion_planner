"""Training loop for conditional diffusion map completion model.

Usage:
    # Local (MPS):
    python src/train.py --train_dir data/train --val_dir data/val --device mps --batch_size 8

    # AWS GPU:
    python src/train.py --train_dir data/train --val_dir data/val --device cuda --batch_size 32

    # With W&B:
    python src/train.py --train_dir data/train --val_dir data/val --device cuda --batch_size 32 --wandb
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from artifacts import save_denoising_gif, save_diversity_grid, save_uncertainty_map
from dataset import MapCompletionDataset
from diffusion import DDPMScheduler
from unet import ConditionalUNet


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "gifs"), exist_ok=True)

    # W&B setup
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="diffusion-map-completion",
            name=args.wandb_name or f"run_{time.strftime('%m%d_%H%M')}",
            config=vars(args),
            save_code=True,
        )

    print(f"Device: {device}")
    print(f"Loading training data from {args.train_dir}...")
    train_ds = MapCompletionDataset(args.train_dir)
    print(f"  {len(train_ds)} training samples")

    val_ds = None
    if args.val_dir and Path(args.val_dir).exists():
        val_ds = MapCompletionDataset(args.val_dir)
        print(f"  {len(val_ds)} validation samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )

    model = ConditionalUNet(
        in_channels=3, out_channels=1,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        time_dim=args.time_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    scheduler = DDPMScheduler(num_timesteps=args.T, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch}")

    history = {"train_loss": [], "val_loss": [], "epoch": [], "iou": []}

    scaler = None
    use_amp = device.type == "cuda"
    if use_amp:
        scaler = torch.amp.GradScaler()

    global_step = start_epoch * min(args.max_steps, len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if args.max_steps and batch_idx >= args.max_steps:
                break

            partial_map = batch["partial_map"].to(device)
            known_mask = batch["known_mask"].to(device)
            full_map = batch["full_map"].to(device)

            t = torch.randint(0, args.T, (full_map.shape[0],), device=device)
            x_noisy, noise = scheduler.q_sample(full_map, t)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred_noise = model(x_noisy, t, partial_map, known_mask)
                    loss = F.mse_loss(pred_noise, noise)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_noise = model(x_noisy, t, partial_map, known_mask)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if wandb_run and batch_idx % 50 == 0:
                wandb_run.log({
                    "train/loss_step": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/global_step": global_step,
                }, step=global_step)

            if batch_idx % 200 == 0:
                print(f"  [{batch_idx}/{min(args.max_steps or len(train_loader), len(train_loader))}] "
                      f"loss: {loss.item():.5f}")

        lr_sched.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        val_loss = None
        val_iou = None
        if val_loader and (epoch + 1) % args.val_every == 0:
            val_loss, val_iou = validate(model, scheduler, val_loader, device, args.T)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["iou"].append(val_iou)

        vstr = f" | val_loss: {val_loss:.5f}" if val_loss is not None else ""
        istr = f" | val_iou: {val_iou:.3f}" if val_iou is not None else ""
        print(f"Epoch {epoch+1}/{args.epochs} | loss: {avg_train_loss:.5f}{vstr}{istr}"
              f" | lr: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if wandb_run:
            log = {"train/loss_epoch": avg_train_loss, "train/epoch": epoch + 1}
            if val_loss is not None:
                log["val/loss"] = val_loss
            if val_iou is not None:
                log["val/iou"] = val_iou
            wandb_run.log(log, step=global_step)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.ckpt_dir, f"model_epoch{epoch+1:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)

        if (epoch + 1) % args.sample_every == 0:
            sample_imgs = generate_samples(model, scheduler, val_loader or train_loader,
                                           device, epoch, args.log_dir)

            loader = val_loader or train_loader
            batch = next(iter(loader))
            pm = batch["partial_map"][:1].to(device)
            km = batch["known_mask"][:1].to(device)
            fm = batch["full_map"][:1].to(device)

            gif_dir = os.path.join(args.log_dir, "gifs")
            try:
                gif_path = os.path.join(gif_dir, f"denoise_epoch{epoch+1:04d}.gif")
                save_denoising_gif(model, scheduler, pm, km, device, gif_path)

                div_path = os.path.join(args.log_dir, "samples",
                                        f"diversity_epoch{epoch+1:04d}.png")
                save_diversity_grid(model, scheduler, pm, km, fm, device, div_path)

                unc_path = os.path.join(args.log_dir, "samples",
                                        f"uncertainty_epoch{epoch+1:04d}.png")
                save_uncertainty_map(model, scheduler, pm, km, device, unc_path)

                if wandb_run:
                    import wandb as wb
                    wandb_run.log({
                        "samples/predictions": wb.Image(sample_imgs),
                        "samples/diversity": wb.Image(div_path),
                        "samples/uncertainty": wb.Image(unc_path),
                        "samples/denoising_gif": wb.Video(gif_path, fps=10, format="gif"),
                    }, step=global_step)
            except Exception as e:
                print(f"Artifact generation failed: {e}")

        with open(os.path.join(args.log_dir, "history.json"), "w") as f:
            json.dump(history, f)
        plot_loss(history, os.path.join(args.log_dir, "loss_curve.png"))

    ckpt_path = os.path.join(args.ckpt_dir, "model_final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, ckpt_path)
    print(f"Training complete. Final model saved to {ckpt_path}")

    if wandb_run:
        import wandb as wb
        wandb_run.log({"results/loss_curve": wb.Image(
            os.path.join(args.log_dir, "loss_curve.png"))}, step=global_step)
        artifact = wb.Artifact("model-final", type="model")
        artifact.add_file(ckpt_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


@torch.no_grad()
def validate(model, scheduler, val_loader, device, T):
    model.eval()
    vloss = 0.0
    vn = 0
    ious = []

    for batch in val_loader:
        partial_map = batch["partial_map"].to(device)
        known_mask = batch["known_mask"].to(device)
        full_map = batch["full_map"].to(device)

        t = torch.randint(0, T, (full_map.shape[0],), device=device)
        x_noisy, noise = scheduler.q_sample(full_map, t)
        pred = model(x_noisy, t, partial_map, known_mask)
        vloss += F.mse_loss(pred, noise).item()
        vn += 1

    if vn > 0:
        pred_maps = scheduler.sample_ddim(model, partial_map[:4], known_mask[:4], num_steps=50)
        for i in range(min(4, pred_maps.shape[0])):
            gt = (full_map[i, 0].cpu().numpy() + 1) / 2
            p = (pred_maps[i, 0].cpu().numpy() + 1) / 2
            ious.append(compute_iou(gt, p))

    return vloss / max(vn, 1), np.mean(ious) if ious else None


@torch.no_grad()
def generate_samples(model, scheduler, loader, device, epoch, log_dir, num_samples=4):
    model.eval()
    batch = next(iter(loader))
    partial_map = batch["partial_map"][:num_samples].to(device)
    known_mask = batch["known_mask"][:num_samples].to(device)
    full_map = batch["full_map"][:num_samples].to(device)

    predicted = scheduler.sample_ddim(model, partial_map, known_mask, num_steps=50)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        gt = (full_map[i, 0].cpu().numpy() + 1) / 2
        pm = (partial_map[i, 0].cpu().numpy() + 1) / 2
        km = known_mask[i, 0].cpu().numpy()
        pred = (predicted[i, 0].cpu().numpy() + 1) / 2

        axes[i, 0].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis("off")

        vis = np.zeros((*pm.shape, 3))
        vis[pm > 0.6] = [1, 1, 1]
        vis[pm < 0.4] = [0, 0, 0]
        vis[(pm >= 0.4) & (pm <= 0.6)] = [0.5, 0.5, 0.7]
        axes[i, 1].imshow(vis)
        axes[i, 1].set_title(f"Partial ({km.mean():.0%} known)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred.clip(0, 1), cmap="gray_r", vmin=0, vmax=1)
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")

        error = np.abs(gt - pred.clip(0, 1))
        axes[i, 3].imshow(error, cmap="hot", vmin=0, vmax=1)
        axes[i, 3].set_title(f"Error (IoU: {compute_iou(gt, pred):.2f})")
        axes[i, 3].axis("off")

    plt.suptitle(f"Epoch {epoch + 1}")
    plt.tight_layout()
    save_path = os.path.join(log_dir, f"samples_epoch{epoch+1:04d}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


def compute_iou(gt, pred, threshold=0.5):
    gt_bin = gt > threshold
    pred_bin = pred > threshold
    intersection = (gt_bin & pred_bin).sum()
    union = (gt_bin | pred_bin).sum()
    return intersection / max(union, 1)


def plot_loss(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train", color="blue", linewidth=2)
    val = [(e, v) for e, v in zip(history["epoch"], history["val_loss"]) if v is not None]
    if val:
        axes[0].plot(*zip(*val), label="Val", color="orange", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ious = [(e, v) for e, v in zip(history["epoch"], history.get("iou", [])) if v is not None]
    if ious:
        axes[1].plot(*zip(*ious), label="Val IoU", color="green", linewidth=2, marker="o", markersize=4)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("IoU")
        axes[1].set_title("Map Prediction IoU")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "IoU computed\nstarting epoch 2", ha="center", va="center",
                     fontsize=14, color="gray", transform=axes[1].transAxes)
        axes[1].set_title("Map Prediction IoU")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 4, 4])
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Max batches per epoch (caps epoch length for large datasets)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()
    train(args)
