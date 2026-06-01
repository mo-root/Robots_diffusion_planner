"""Generate plots from ablation, realtime, and latency experiments.

Reads (any subset present):
  results/experiments_ablation/k_sweep.json
  results/experiments_ablation/lam_sweep.json
  results/experiments_realtime/realtime_results.json
  results/latency_profile.json

Writes figures into results/analysis/figures/:
  05_ablation_k.png
  05b_ablation_lambda.png
  06_realtime_compare.png
  06b_latency_profile.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("results/analysis/figures")
FIG.mkdir(parents=True, exist_ok=True)

COL_DIFF = "#c576ff"
COL_BASE = "#ffb86c"
COL_GOOD = "#7ee787"
COL_ACC = "#6fb6ff"

plt.rcParams.update({
    "figure.facecolor": "#0c1019",
    "axes.facecolor": "#0c1019",
    "axes.edgecolor": "#3a4258",
    "axes.labelcolor": "#dfe6f0",
    "xtick.color": "#dfe6f0",
    "ytick.color": "#8d97a8",
    "text.color": "#dfe6f0",
    "axes.grid": True,
    "grid.color": "#1a2030",
    "grid.linewidth": 0.5,
    "font.size": 12,
})


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def plot_k_sweep():
    d = load("results/experiments_ablation/k_sweep.json")
    if not d:
        print("(skip K-sweep: no data)"); return
    ks = sorted(int(k) for k in d.keys())
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for k in ks:
        runs = d[str(k)]
        if not runs: continue
        # average per-step coverage
        maxlen = max(len(r["coverage"]) for r in runs)
        steps = []
        for s in range(maxlen):
            vals = [r["coverage"][s] if s < len(r["coverage"]) else r["coverage"][-1] for r in runs]
            steps.append(np.mean(vals))
        x = np.arange(1, len(steps) + 1)
        color = plt.cm.plasma(np.log2(k) / np.log2(max(ks))) if k > 1 else "#888"
        ax.plot(x, steps, color=color, linewidth=2.2, label=f"K = {k}  (n={len(runs)})")
    ax.set_xlabel("step")
    ax.set_ylabel("mean coverage")
    ax.set_title("K-sample ablation: does drawing more completions help?", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142")
    fig.tight_layout()
    out = FIG / "05_ablation_k.png"
    fig.savefig(out, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"wrote {out}")


def plot_lambda_sweep():
    d = load("results/experiments_ablation/lam_sweep.json")
    if not d:
        print("(skip lambda sweep: no data)"); return
    lams = sorted(float(l) for l in d.keys())
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for lam in lams:
        runs = d[str(lam)]
        if not runs: continue
        maxlen = max(len(r["coverage"]) for r in runs)
        steps = []
        for s in range(maxlen):
            vals = [r["coverage"][s] if s < len(r["coverage"]) else r["coverage"][-1] for r in runs]
            steps.append(np.mean(vals))
        x = np.arange(1, len(steps) + 1)
        color = plt.cm.viridis(lam / max(lams)) if max(lams) > 0 else "#888"
        ax.plot(x, steps, color=color, linewidth=2.2, label=f"λ = {lam}  (n={len(runs)})")
    ax.set_xlabel("step")
    ax.set_ylabel("mean coverage")
    ax.set_title("Lambda ablation: how much does K-sample uncertainty matter?", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142")
    fig.tight_layout()
    out = FIG / "05b_ablation_lambda.png"
    fig.savefig(out, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"wrote {out}")


def plot_realtime():
    d = load("results/experiments_realtime/realtime_results.json")
    if not d:
        print("(skip realtime: no data)"); return
    if not d: return
    maxlen = max(len(r["base_cov"]) for r in d)
    def stats(key):
        out_m, out_se = [], []
        for s in range(maxlen):
            vals = [r[key][s] if s < len(r[key]) else r[key][-1] for r in d]
            out_m.append(np.mean(vals))
            out_se.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        return np.array(out_m), np.array(out_se)

    base_m, base_se = stats("base_cov")
    disc_m, disc_se = stats("disc_cov")
    rt_m, rt_se = stats("rt_cov")

    x = np.arange(1, maxlen + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for m, se, color, label in [(base_m, base_se, COL_BASE, "baseline"),
                                 (disc_m, disc_se, COL_DIFF, "diffusion (per-decision)"),
                                 (rt_m, rt_se, COL_GOOD, "diffusion (realtime re-plan)")]:
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.16)
        ax.plot(x, m, color=color, linewidth=2.6, label=label)
    ax.set_xlabel("frontier-decision step")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1)
    ax.set_title(f"Realtime re-planning vs per-decision (N={len(d)})", fontsize=13)
    ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142")
    fig.tight_layout()
    out = FIG / "06_realtime_compare.png"
    fig.savefig(out, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"wrote {out}")
    # report headline numbers
    if len(rt_m) > 3:
        print(f"  step-4:   rt={rt_m[3]:.3f}  disc={disc_m[3]:.3f}  base={base_m[3]:.3f}")
        print(f"  rt Δ vs base @ step4: {(rt_m[3]-base_m[3])*100:+.2f}%")
        print(f"  rt Δ vs disc @ step4: {(rt_m[3]-disc_m[3])*100:+.2f}%")


def plot_latency():
    d = load("results/latency_profile.json")
    if not d:
        print("(skip latency: no data)"); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=120)

    # K vs latency
    K = sorted(int(k) for k in d["K_latency_ms"].keys())
    K_lat = [d["K_latency_ms"][str(k)] for k in K]
    axes[0].plot(K, K_lat, color=COL_ACC, marker="o", linewidth=2.4, markersize=10)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(K); axes[0].set_xticklabels([str(k) for k in K])
    axes[0].set_xlabel("K (number of completions)")
    axes[0].set_ylabel("latency per call (ms)")
    axes[0].set_title("Inference latency vs K", fontsize=12)
    for k, l in zip(K, K_lat):
        axes[0].annotate(f"{l:.0f}ms", (k, l), xytext=(5, 5), textcoords="offset points", fontsize=10)

    # DDIM step vs latency
    steps = sorted(int(s) for s in d["ddim_step_latency_ms"].keys())
    s_lat = [d["ddim_step_latency_ms"][str(s)] for s in steps]
    axes[1].plot(steps, s_lat, color=COL_DIFF, marker="o", linewidth=2.4, markersize=10)
    axes[1].set_xlabel("DDIM sampling steps")
    axes[1].set_ylabel("latency per call (ms)")
    axes[1].set_title("Inference latency vs DDIM steps", fontsize=12)
    for s, l in zip(steps, s_lat):
        axes[1].annotate(f"{l:.0f}ms", (s, l), xytext=(5, 5), textcoords="offset points", fontsize=10)

    fig.suptitle("Where the budget goes (T4 GPU)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = FIG / "06b_latency_profile.png"
    fig.savefig(out, dpi=120, facecolor="#0c1019", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_k_sweep()
    plot_lambda_sweep()
    plot_realtime()
    plot_latency()
