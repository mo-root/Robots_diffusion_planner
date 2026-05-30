"""Analyze and plot experiment results from experiments.py.

Produces publication-quality figures:
- Area-over-time curves with 95% CIs (Exp 1)
- Per-step win-rate (Exp 1)
- Steps-to-coverage thresholds histogram (Exp 1)
- Lidar range advantage chart (Exp 3)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "diffusion": "#2E7D32",
    "baseline": "#E65100",
    "target": "#C62828",
    "accent": "#1565C0",
}


def pad_curves(curves, max_len):
    """Pad each curve to max_len by holding the last value."""
    padded = []
    for c in curves:
        if len(c) >= max_len:
            padded.append(c[:max_len])
        else:
            padded.append(c + [c[-1]] * (max_len - len(c)))
    return np.array(padded)


def plot_exp1_area_over_time(results, out_path):
    max_len = max(max(len(r["diff_cov"]), len(r["base_cov"])) for r in results)
    diff = pad_curves([r["diff_cov"] for r in results], max_len)
    base = pad_curves([r["base_cov"] for r in results], max_len)
    steps = np.arange(1, max_len + 1)

    diff_mean = diff.mean(axis=0)
    base_mean = base.mean(axis=0)
    diff_se = diff.std(axis=0) / np.sqrt(len(diff))
    base_se = base.std(axis=0) / np.sqrt(len(base))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, diff_mean, "o-", color=COLORS["diffusion"], linewidth=2.5,
            markersize=7, label=f"Diffusion (ours), N={len(results)}")
    ax.fill_between(steps, diff_mean - 1.96*diff_se, diff_mean + 1.96*diff_se,
                    color=COLORS["diffusion"], alpha=0.2, label="95% CI")
    ax.plot(steps, base_mean, "s-", color=COLORS["baseline"], linewidth=2.5,
            markersize=7, label=f"Baseline heuristic, N={len(results)}")
    ax.fill_between(steps, base_mean - 1.96*base_se, base_mean + 1.96*base_se,
                    color=COLORS["baseline"], alpha=0.2)
    ax.axhline(y=0.8, color=COLORS["target"], linestyle="--", alpha=0.6, label="80% target")

    ax.set_xlabel("Exploration step", fontsize=13)
    ax.set_ylabel("Area explored (fraction of free space)", fontsize=13)
    ax.set_title(f"Area Explored Over Time (mean ± 95% CI, N={len(results)} maps)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.5, max_len + 0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")
    return diff_mean, base_mean


def plot_exp1_winrate_per_step(results, out_path):
    max_len = max(max(len(r["diff_cov"]), len(r["base_cov"])) for r in results)
    diff = pad_curves([r["diff_cov"] for r in results], max_len)
    base = pad_curves([r["base_cov"] for r in results], max_len)
    steps = np.arange(1, max_len + 1)

    diff_wins = (diff > base + 0.005).sum(axis=0) / len(diff)
    base_wins = (base > diff + 0.005).sum(axis=0) / len(diff)
    ties = 1 - diff_wins - base_wins

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(steps - 0.27, diff_wins, width=0.27, label="Diffusion wins",
           color=COLORS["diffusion"], alpha=0.9)
    ax.bar(steps, ties, width=0.27, label="Tie (<0.5% diff)", color="#9E9E9E", alpha=0.7)
    ax.bar(steps + 0.27, base_wins, width=0.27, label="Baseline wins",
           color=COLORS["baseline"], alpha=0.9)
    ax.set_xlabel("Exploration step", fontsize=13)
    ax.set_ylabel("Fraction of maps", fontsize=13)
    ax.set_title(f"Per-Step Win Rate (N={len(results)} maps)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_exp1_steps_to_threshold(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, thresh, key_d, key_b in [
        (axes[0], "80%", "diff_to_80", "base_to_80"),
        (axes[1], "90%", "diff_to_90", "base_to_90"),
    ]:
        d = [r[key_d] for r in results if r[key_d] > 0]
        b = [r[key_b] for r in results if r[key_b] > 0]
        max_step = max(max(d, default=0), max(b, default=0)) + 1
        bins = np.arange(0.5, max_step + 1, 1)
        ax.hist(d, bins=bins, alpha=0.6, label=f"Diffusion (mean={np.mean(d):.1f})",
                color=COLORS["diffusion"], edgecolor="black")
        ax.hist(b, bins=bins, alpha=0.6, label=f"Baseline (mean={np.mean(b):.1f})",
                color=COLORS["baseline"], edgecolor="black")
        ax.set_xlabel("Steps to reach threshold", fontsize=12)
        ax.set_ylabel("Number of maps", fontsize=12)
        ax.set_title(f"Steps to {thresh} coverage", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_exp1_per_map_scatter(results, out_path):
    diff_80 = np.array([r["diff_to_80"] for r in results])
    base_80 = np.array([r["base_to_80"] for r in results])
    mask = (diff_80 > 0) & (base_80 > 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(base_80[mask], diff_80[mask], s=80, alpha=0.7,
               c=COLORS["diffusion"], edgecolors="black")
    lo = min(diff_80[mask].min(), base_80[mask].min())
    hi = max(diff_80[mask].max(), base_80[mask].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Tie line")

    diff_faster = (diff_80[mask] < base_80[mask]).sum()
    base_faster = (base_80[mask] < diff_80[mask]).sum()
    ties = (diff_80[mask] == base_80[mask]).sum()

    ax.text(0.05, 0.95, f"Diffusion faster: {diff_faster}\n"
                       f"Baseline faster: {base_faster}\n"
                       f"Tie: {ties}",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("Baseline: steps to 80%", fontsize=12)
    ax.set_ylabel("Diffusion: steps to 80%", fontsize=12)
    ax.set_title("Per-Map Comparison: Steps to 80% Coverage\n(below diagonal = diffusion wins)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_exp3_lidar_sweep(results, out_path):
    """Show how diffusion advantage changes with lidar range."""
    by_lidar = {}
    for r in results:
        by_lidar.setdefault(r["lidar"], []).append(r)

    lidars = sorted(by_lidar.keys())
    diff_means = []
    base_means = []
    diff_se = []
    base_se = []
    advantage_means = []
    advantage_se = []
    diff_to_80 = []
    base_to_80 = []

    for lr in lidars:
        rs = by_lidar[lr]
        d_f = np.array([r["diff_final"] for r in rs])
        b_f = np.array([r["base_final"] for r in rs])
        diff_means.append(d_f.mean())
        base_means.append(b_f.mean())
        diff_se.append(d_f.std() / np.sqrt(len(d_f)))
        base_se.append(b_f.std() / np.sqrt(len(b_f)))

        adv = d_f - b_f
        advantage_means.append(adv.mean())
        advantage_se.append(adv.std() / np.sqrt(len(adv)))

        d80 = [r["diff_to_80"] for r in rs if r["diff_to_80"] > 0]
        b80 = [r["base_to_80"] for r in rs if r["base_to_80"] > 0]
        diff_to_80.append(np.mean(d80) if d80 else np.nan)
        base_to_80.append(np.mean(b80) if b80 else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.errorbar(lidars, diff_means, yerr=[1.96*s for s in diff_se], fmt="o-",
                color=COLORS["diffusion"], linewidth=2.5, markersize=10,
                label="Diffusion (ours)", capsize=5)
    ax.errorbar(lidars, base_means, yerr=[1.96*s for s in base_se], fmt="s-",
                color=COLORS["baseline"], linewidth=2.5, markersize=10,
                label="Baseline heuristic", capsize=5)
    ax.set_xlabel("Lidar range (pixels)", fontsize=12)
    ax.set_ylabel("Final coverage (after 20 steps)", fontsize=12)
    ax.set_title("Final Coverage vs Sensor Range", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(lidars, advantage_means, yerr=[1.96*s for s in advantage_se],
           width=8, color=COLORS["accent"], edgecolor="black", capsize=5, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Lidar range (pixels)", fontsize=12)
    ax.set_ylabel("Diffusion advantage (coverage delta)", fontsize=12)
    ax.set_title("Where Does Prediction Help Most?\n(positive = diffusion wins)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot(lidars, diff_to_80, "o-", color=COLORS["diffusion"], linewidth=2.5,
            markersize=10, label="Diffusion")
    ax.plot(lidars, base_to_80, "s-", color=COLORS["baseline"], linewidth=2.5,
            markersize=10, label="Baseline")
    ax.set_xlabel("Lidar range (pixels)", fontsize=12)
    ax.set_ylabel("Mean steps to 80%", fontsize=12)
    ax.set_title("Speed to 80% Coverage", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_exp3_curves_per_lidar(results, out_path):
    by_lidar = {}
    for r in results:
        by_lidar.setdefault(r["lidar"], []).append(r)

    lidars = sorted(by_lidar.keys())
    fig, axes = plt.subplots(1, len(lidars), figsize=(5.5 * len(lidars), 5), sharey=True)
    if len(lidars) == 1:
        axes = [axes]

    for ax, lr in zip(axes, lidars):
        rs = by_lidar[lr]
        max_len = max(max(len(r["diff_cov"]), len(r["base_cov"])) for r in rs)
        diff = pad_curves([r["diff_cov"] for r in rs], max_len)
        base = pad_curves([r["base_cov"] for r in rs], max_len)
        steps = np.arange(1, max_len + 1)

        d_mean, d_se = diff.mean(0), diff.std(0)/np.sqrt(len(diff))
        b_mean, b_se = base.mean(0), base.std(0)/np.sqrt(len(base))

        ax.plot(steps, d_mean, "o-", color=COLORS["diffusion"], linewidth=2, label="Diffusion")
        ax.fill_between(steps, d_mean-1.96*d_se, d_mean+1.96*d_se,
                        color=COLORS["diffusion"], alpha=0.2)
        ax.plot(steps, b_mean, "s-", color=COLORS["baseline"], linewidth=2, label="Baseline")
        ax.fill_between(steps, b_mean-1.96*b_se, b_mean+1.96*b_se,
                        color=COLORS["baseline"], alpha=0.2)
        ax.axhline(y=0.8, color=COLORS["target"], linestyle="--", alpha=0.5)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_title(f"Lidar = {lr} px", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.02)
        if ax is axes[0]:
            ax.set_ylabel("Coverage", fontsize=12)
            ax.legend(fontsize=10, loc="lower right")

    plt.suptitle(f"Area-Over-Time by Sensor Range (N={len(by_lidar[lidars[0]])} maps each)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def write_summary(exp1_results, exp3_results, out_path):
    lines = ["# Experiment Summary\n"]
    if exp1_results:
        lines.append(f"## Experiment 1: N={len(exp1_results)} maps, lidar=70\n")
        d_final = np.mean([r["diff_final"] for r in exp1_results])
        b_final = np.mean([r["base_final"] for r in exp1_results])
        d80 = [r["diff_to_80"] for r in exp1_results if r["diff_to_80"] > 0]
        b80 = [r["base_to_80"] for r in exp1_results if r["base_to_80"] > 0]
        d90 = [r["diff_to_90"] for r in exp1_results if r["diff_to_90"] > 0]
        b90 = [r["base_to_90"] for r in exp1_results if r["base_to_90"] > 0]
        lines.append(f"- Final coverage: diff={d_final:.1%}, base={b_final:.1%}\n")
        lines.append(f"- Mean steps to 80%: diff={np.mean(d80):.2f} (n={len(d80)}), "
                     f"base={np.mean(b80):.2f} (n={len(b80)})\n")
        lines.append(f"- Mean steps to 90%: diff={np.mean(d90):.2f} (n={len(d90)}), "
                     f"base={np.mean(b90):.2f} (n={len(b90)})\n")
        d80a = np.array([r["diff_to_80"] for r in exp1_results])
        b80a = np.array([r["base_to_80"] for r in exp1_results])
        valid = (d80a > 0) & (b80a > 0)
        if valid.sum() > 0:
            d_faster = int((d80a[valid] < b80a[valid]).sum())
            b_faster = int((b80a[valid] < d80a[valid]).sum())
            ties = int((d80a[valid] == b80a[valid]).sum())
            lines.append(f"- Faster to 80%: diffusion={d_faster}/{valid.sum()}, "
                         f"baseline={b_faster}/{valid.sum()}, ties={ties}\n")

    if exp3_results:
        lines.append(f"\n## Experiment 3: lidar range sweep\n")
        by_lidar = {}
        for r in exp3_results:
            by_lidar.setdefault(r["lidar"], []).append(r)
        for lr in sorted(by_lidar.keys()):
            rs = by_lidar[lr]
            d_f = np.mean([r["diff_final"] for r in rs])
            b_f = np.mean([r["base_final"] for r in rs])
            adv = d_f - b_f
            lines.append(f"- Lidar {lr}px: diff={d_f:.1%}, base={b_f:.1%}, "
                         f"advantage={adv:+.1%}\n")

    with open(out_path, "w") as fh:
        fh.writelines(lines)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="results/experiments")
    parser.add_argument("--out_dir", default="results/experiments/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    exp1_path = os.path.join(args.exp_dir, "exp1_results.json")
    exp3_path = os.path.join(args.exp_dir, "exp3_results.json")

    exp1 = None
    if os.path.exists(exp1_path):
        with open(exp1_path) as fh:
            exp1 = json.load(fh)
        print(f"Exp 1: {len(exp1)} maps")
        plot_exp1_area_over_time(exp1, os.path.join(args.out_dir, "exp1_area_over_time.png"))
        plot_exp1_winrate_per_step(exp1, os.path.join(args.out_dir, "exp1_winrate.png"))
        plot_exp1_steps_to_threshold(exp1, os.path.join(args.out_dir, "exp1_steps_to_threshold.png"))
        plot_exp1_per_map_scatter(exp1, os.path.join(args.out_dir, "exp1_scatter.png"))

    exp3 = None
    if os.path.exists(exp3_path):
        with open(exp3_path) as fh:
            exp3 = json.load(fh)
        print(f"Exp 3: {len(exp3)} runs")
        plot_exp3_lidar_sweep(exp3, os.path.join(args.out_dir, "exp3_lidar_sweep.png"))
        plot_exp3_curves_per_lidar(exp3, os.path.join(args.out_dir, "exp3_curves_per_lidar.png"))

    if exp1 or exp3:
        write_summary(exp1 or [], exp3 or [], os.path.join(args.out_dir, "summary.md"))


if __name__ == "__main__":
    main()
