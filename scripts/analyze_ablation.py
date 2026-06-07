#!/usr/bin/env python3
"""Analyze 4-condition ablation results from run_ablation.sh.

For each method × trial JSON file produced by the exploration_manager,
compute:
  - mean / median info-gain per scoring decision
  - time-to-X%-coverage (X in {50, 75, 85})
  - head-to-head record (wins / ties / losses) of diffusion vs each baseline
  - coverage curves (steps → coverage)

Outputs:
  results/ablation/summary.csv         — per-method aggregate metrics
  results/ablation/coverage_curves.png — coverage vs steps, all methods
  results/ablation/info_gain_bar.png   — mean info-gain bar chart
  results/ablation/head_to_head.csv    — W/T/L vs diffusion

Trial JSON format expected (each line is one scoring decision):
  {"t": 12.3, "step": 5, "coverage": 0.34, "chosen_gain": 47.0,
   "method": "diffusion", "trial": 3}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ["nearest", "info_gain", "heuristic", "diffusion"]
COLORS = {
    "nearest":   "#f85149",
    "info_gain": "#d29922",
    "heuristic": "#58a6ff",
    "diffusion": "#3fb950",
}


def load_trial(path: Path) -> list[dict]:
    """Each JSON file is newline-delimited dicts, one per scoring decision."""
    decisions = []
    if not path.exists():
        return decisions
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return decisions


def trial_summary(decisions: list[dict]) -> dict:
    if not decisions:
        return {
            "n_decisions": 0,
            "final_coverage": 0.0,
            "mean_gain": 0.0,
            "median_gain": 0.0,
            "time_to_50": None, "time_to_75": None, "time_to_85": None,
        }
    gains = np.array([d.get("chosen_gain", 0.0) for d in decisions])
    coverage = np.array([d.get("coverage", 0.0) for d in decisions])
    times = np.array([d.get("t", float(i))
                      for i, d in enumerate(decisions)])

    def time_to(threshold: float) -> float | None:
        idx = np.argmax(coverage >= threshold) if (coverage >= threshold).any() else -1
        return float(times[idx]) if idx >= 0 and coverage[idx] >= threshold else None

    return {
        "n_decisions":    len(decisions),
        "final_coverage": float(coverage[-1]),
        "mean_gain":      float(np.mean(gains)),
        "median_gain":    float(np.median(gains)),
        "time_to_50":     time_to(0.50),
        "time_to_75":     time_to(0.75),
        "time_to_85":     time_to(0.85),
    }


def aggregate(method_dir: Path) -> dict:
    """Aggregate all trials for one method."""
    trial_files = sorted(method_dir.glob("trial_*.json"))
    rows = [trial_summary(load_trial(p)) for p in trial_files]
    if not rows:
        return {}

    def col(k):
        vals = [r[k] for r in rows if r[k] is not None]
        return np.array(vals) if vals else np.array([0.0])

    return {
        "n_trials":         len(rows),
        "mean_gain":        float(np.mean(col("mean_gain"))),
        "std_gain":         float(np.std(col("mean_gain"))),
        "mean_final_cov":   float(np.mean(col("final_coverage"))),
        "mean_t_to_50":     float(np.mean(col("time_to_50"))) if col("time_to_50").size else None,
        "mean_t_to_75":     float(np.mean(col("time_to_75"))) if col("time_to_75").size else None,
        "mean_t_to_85":     float(np.mean(col("time_to_85"))) if col("time_to_85").size else None,
        "per_trial_gain":   col("mean_gain").tolist(),
        "trial_files":      [str(p.name) for p in trial_files],
    }


def write_summary_csv(per_method: dict, out_path: Path) -> None:
    fields = ["method", "n_trials", "mean_gain", "std_gain",
              "mean_final_cov", "mean_t_to_50", "mean_t_to_75", "mean_t_to_85"]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for m in METHODS:
            r = per_method.get(m, {})
            if not r:
                continue
            w.writerow([m, r["n_trials"], f"{r['mean_gain']:.2f}",
                        f"{r['std_gain']:.2f}", f"{r['mean_final_cov']:.3f}",
                        f"{r['mean_t_to_50']}" if r["mean_t_to_50"] else "-",
                        f"{r['mean_t_to_75']}" if r["mean_t_to_75"] else "-",
                        f"{r['mean_t_to_85']}" if r["mean_t_to_85"] else "-"])


def head_to_head(per_method: dict, out_path: Path) -> None:
    """For each baseline, count W/T/L of diffusion vs it across paired trials."""
    diff_trials = per_method.get("diffusion", {}).get("per_trial_gain", [])
    rows = [["baseline", "diffusion_wins", "ties", "diffusion_losses"]]
    for m in METHODS:
        if m == "diffusion":
            continue
        base_trials = per_method.get(m, {}).get("per_trial_gain", [])
        n = min(len(diff_trials), len(base_trials))
        if n == 0:
            rows.append([m, 0, 0, 0]); continue
        wins = ties = losses = 0
        for d, b in zip(diff_trials[:n], base_trials[:n]):
            if d > b + 0.5:    wins   += 1
            elif d < b - 0.5:  losses += 1
            else:              ties   += 1
        rows.append([m, wins, ties, losses])
    with out_path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def plot_gain_bar(per_method: dict, out_path: Path) -> None:
    methods = [m for m in METHODS if m in per_method]
    means = [per_method[m]["mean_gain"] for m in methods]
    stds = [per_method[m]["std_gain"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, color=colors, capsize=6,
           edgecolor="#1c2128", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=11)
    ax.set_ylabel("Mean info-gain at chosen frontier", fontsize=11)
    ax.set_title("Frontier scoring — 4-condition ablation", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 2, f"{m:.1f}", ha="center", fontsize=10,
                color="#1c2128")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_coverage_curves(ablation_dir: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in METHODS:
        method_dir = ablation_dir / m
        if not method_dir.exists():
            continue
        # plot each trial as a faint line + mean as a bold line
        all_curves = []
        for tf in sorted(method_dir.glob("trial_*.json")):
            decisions = load_trial(tf)
            if not decisions:
                continue
            cov = [d.get("coverage", 0.0) for d in decisions]
            ax.plot(range(len(cov)), cov, color=COLORS[m], alpha=0.15, linewidth=1)
            all_curves.append(cov)
        if all_curves:
            # pad to longest, then mean
            L = max(len(c) for c in all_curves)
            padded = np.array([c + [c[-1]] * (L - len(c)) for c in all_curves])
            ax.plot(range(L), padded.mean(axis=0), color=COLORS[m],
                    label=m, linewidth=2.4)
    ax.set_xlabel("Scoring decision (step)", fontsize=11)
    ax.set_ylabel("Map coverage (fraction)", fontsize=11)
    ax.set_title("Coverage progression — 4-condition ablation", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation_dir", required=True,
                    help="Top-level dir holding {nearest,info_gain,heuristic,diffusion}/trial_*.json")
    args = ap.parse_args()

    ab_dir = Path(args.ablation_dir)
    if not ab_dir.exists():
        print(f"error: {ab_dir} does not exist", file=sys.stderr); sys.exit(1)

    per_method = {m: aggregate(ab_dir / m) for m in METHODS}
    per_method = {k: v for k, v in per_method.items() if v}

    write_summary_csv(per_method, ab_dir / "summary.csv")
    head_to_head(per_method, ab_dir / "head_to_head.csv")
    plot_gain_bar(per_method, ab_dir / "info_gain_bar.png")
    plot_coverage_curves(ab_dir, ab_dir / "coverage_curves.png")

    print(f"\nresults written to {ab_dir}/")
    print(f"  - summary.csv          (per-method aggregates)")
    print(f"  - head_to_head.csv     (W/T/L of diffusion vs each baseline)")
    print(f"  - info_gain_bar.png    (bar chart for the report)")
    print(f"  - coverage_curves.png  (coverage progression plot)")

    # Console summary
    print("\nper-method summary:")
    print(f"  {'method':<12} {'trials':>7} {'mean_gain':>10} {'final_cov':>10}")
    for m in METHODS:
        r = per_method.get(m, {})
        if not r: continue
        print(f"  {m:<12} {r['n_trials']:>7} {r['mean_gain']:>10.2f} "
              f"{r['mean_final_cov']:>10.3f}")


if __name__ == "__main__":
    main()
