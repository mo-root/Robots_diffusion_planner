"""The 2x2 matrix figure: prior (in-domain vs OOD) × sampling (discrete vs realtime).
Shows that sampling frequency matters more than domain match in our data."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/analysis/figures/11_2x2_matrix.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None

in_rt = load("results/experiments_realtime/realtime_results.json")           # HouseExpo model rollout (rt + disc + base)
ood_rt = load("results/experiments_warehouse_realtime/realtime_results.json") # warehouse model rollout (rt + disc + base)

def step4_mean(records, key):
    vals = [r[key][3] for r in records if len(r[key]) > 3]
    return np.mean(vals) * 100 if vals else None

# All four cells + baseline come from the realtime experiments — same code path,
# same scoring, same random seed. The realtime experiments each ran disc + rt +
# base in the same loop on identical maps, so comparisons within an experiment
# are apples-to-apples.
rt_in   = step4_mean(in_rt, "rt_cov")   if in_rt else None     # in-domain realtime
disc_in = step4_mean(in_rt, "disc_cov") if in_rt else None     # in-domain discrete (from realtime exp's disc baseline)
rt_ood  = step4_mean(ood_rt, "rt_cov")  if ood_rt else None    # OOD realtime
disc_ood= step4_mean(ood_rt, "disc_cov")if ood_rt else None    # OOD discrete (from warehouse-realtime's disc baseline)
base_in = step4_mean(in_rt, "base_cov") if in_rt else None
base_ood= step4_mean(ood_rt, "base_cov")if ood_rt else None

print("Step-4 means:")
print(f"  in-domain realtime:  {rt_in:.2f}%" if rt_in else "  in-domain realtime: ?")
print(f"  in-domain discrete:  {disc_in:.2f}%" if disc_in else "  in-domain discrete: ?")
print(f"  OOD realtime:        {rt_ood:.2f}%" if rt_ood else "  OOD realtime: ?")
print(f"  OOD discrete:        {disc_ood:.2f}%" if disc_ood else "  OOD discrete: ?")
print(f"  baseline (in-domain): {base_in:.2f}%" if base_in else "  baseline: ?")

# absolute deltas vs baseline
b = (base_in + base_ood) / 2 if (base_in and base_ood) else (base_in or base_ood or 62)
deltas = {
    ("in-domain", "discrete"): disc_in - b,
    ("in-domain", "realtime"): rt_in - b,
    ("OOD",       "discrete"): disc_ood - b,
    ("OOD",       "realtime"): rt_ood - b,
}

plt.rcParams.update({
    "figure.facecolor": "#0c1019",
    "axes.facecolor": "#0c1019",
    "axes.edgecolor": "#3a4258",
    "axes.labelcolor": "#dfe6f0",
    "xtick.color": "#dfe6f0",
    "ytick.color": "#dfe6f0",
    "text.color": "#dfe6f0",
    "axes.grid": False,
    "font.size": 12,
})

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)

priors = ["in-domain\n(HouseExpo)", "OOD\n(warehouse)"]
modes  = ["discrete\n(per-decision)", "realtime\n(every 3 cells)"]
x = np.arange(2)
w = 0.36

disc_vals = [deltas[("in-domain", "discrete")], deltas[("OOD", "discrete")]]
rt_vals   = [deltas[("in-domain", "realtime")], deltas[("OOD", "realtime")]]

bars_disc = ax.bar(x - w/2, disc_vals, w, color="#7d6395", edgecolor="#1a2030", linewidth=0.5, label="discrete (per-decision)")
bars_rt   = ax.bar(x + w/2, rt_vals,   w, color="#c576ff", edgecolor="#1a2030", linewidth=0.5, label="realtime (every 3 cells)")

ax.axhline(0, color="#3a4258", linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(priors, fontsize=13)
ax.set_ylabel("step-4 coverage Δ vs baseline (%)", fontsize=13)
ax.set_title("Prior match and inference speed are roughly additive\nSampling frequency matters more than domain match",
             fontsize=13, pad=14, color="#dfe6f0")
ax.legend(loc="upper right", facecolor="#161b26", edgecolor="#2a3142", fontsize=12)
ax.set_ylim(min(-2, min(disc_vals + rt_vals) - 2), max(disc_vals + rt_vals) + 3)

for b in list(bars_disc) + list(bars_rt):
    h = b.get_height()
    sign = "+" if h >= 0 else ""
    ax.text(b.get_x() + b.get_width()/2, h + (0.4 if h >= 0 else -0.6),
            f"{sign}{h:.1f}%",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=14, fontweight="bold",
            color="#c576ff" if "realtime" in str(b) or h > 4 else "#dfe6f0")

# annotate the decomposition with arrows + labels
y_mid = (rt_vals[0] + disc_vals[0]) / 2
ax.annotate("", xy=(0 + w/2, rt_vals[0]), xytext=(0 - w/2, disc_vals[0]),
            arrowprops=dict(arrowstyle="->", color="#6fb6ff", lw=1.5))
ax.annotate("", xy=(1 + w/2, rt_vals[1]), xytext=(1 - w/2, disc_vals[1]),
            arrowprops=dict(arrowstyle="->", color="#6fb6ff", lw=1.5))

fig.tight_layout()
fig.savefig(OUT, dpi=120, facecolor="#0c1019")
plt.close(fig)
print(f"wrote {OUT}")
