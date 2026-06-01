"""Bucket the N=80 complexity-stratified runs into low/mid/high tertiles
by initial frontier count. Plot step-4 advantage per bucket as a bar chart."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("results/experiments_complexity/exp_results.json"))
rows = []
for r in data:
    if "initial_frontiers" not in r:
        continue
    if len(r["diff_cov"]) < 4 or len(r["base_cov"]) < 4:
        continue
    rows.append((r["initial_frontiers"], r["diff_cov"][3] - r["base_cov"][3]))
rows.sort()
n = len(rows)
buckets = {
    "low\n(3–15 frontiers)":  rows[0:n//3],
    "mid\n(15–19 frontiers)": rows[n//3:2*n//3],
    "high\n(19–35 frontiers)": rows[2*n//3:n],
}

labels = list(buckets.keys())
means = [np.mean([d for _, d in buckets[k]]) * 100 for k in labels]
sems = [np.std([d for _, d in buckets[k]], ddof=1) * 100 / np.sqrt(len(buckets[k])) for k in labels]
wins = [sum(1 for _, d in buckets[k] if d > 0.005) for k in labels]
losses = [sum(1 for _, d in buckets[k] if d < -0.005) for k in labels]
ties = [sum(1 for _, d in buckets[k] if abs(d) <= 0.005) for k in labels]
counts = [len(buckets[k]) for k in labels]

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

fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
COL_DIFF = "#c576ff"
x = np.arange(3)
bars = ax.bar(x, means, color=COL_DIFF, width=0.55, edgecolor="#1a2030", linewidth=0.5)
ax.errorbar(x, means, yerr=[1.96 * s for s in sems], fmt="none", ecolor="#dfe6f0", capsize=6, linewidth=1.5)
ax.axhline(0, color="#3a4258", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("step-4 Δ coverage  (diffusion − baseline)", fontsize=13)
ax.set_title("Diffusion advantage grows with map complexity", fontsize=14, color="#dfe6f0", pad=14)
ax.set_ylim(min(-1, min(means) - 2), max(means) + 2.5)

# annotate each bar
for i, (b, m, w, l, t, c) in enumerate(zip(bars, means, wins, losses, ties, counts)):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.4, f"+{m:.2f}%",
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="#c576ff")
    ax.text(b.get_x() + b.get_width() / 2, -0.5,
            f"{w} win  ·  {l} lose  ·  {t} tie\n(n = {c})",
            ha="center", va="top", fontsize=10, color="#8d97a8")

fig.tight_layout()
out = Path("results/analysis/figures/04_complexity_buckets.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, facecolor="#0c1019")
plt.close(fig)
print(f"wrote {out}")
