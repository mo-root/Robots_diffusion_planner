"""Produce the 4-row comparison table + figures from experiments_4baseline/results.json."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results/experiments_4baseline/results.json")
OUT_DIR = Path("results/analysis/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["nearest", "info_gain", "heuristic", "diffusion"]
LABELS = {
    "nearest": "Nearest",
    "info_gain": "Info-gain only",
    "heuristic": "Info-gain + dist",
    "diffusion": "Diffusion (ours)",
}
COLORS = {
    "nearest": "#aab1c0",
    "info_gain": "#7d6395",
    "heuristic": "#6fb6ff",
    "diffusion": "#c576ff",
}

rows = json.loads(RESULTS.read_text())
print(f"Loaded {len(rows)} maps")

MAX_STEPS = 20


def pad_to(cov, n):
    if cov is None or len(cov) == 0:
        return [0.0] * n
    out = list(cov) + [cov[-1]] * (n - len(cov))
    return out[:n]


def summarize(method):
    covs = [pad_to(r[f"{method}_cov"], MAX_STEPS) for r in rows if r.get(f"{method}_cov")]
    arr = np.array(covs) * 100
    final = np.mean([r[f"{method}_final"] for r in rows if f"{method}_final" in r]) * 100
    step4 = np.mean([r[f"{method}_step4"] for r in rows if f"{method}_step4" in r]) * 100
    to80 = [r[f"{method}_to_80"] for r in rows if r.get(f"{method}_to_80", -1) > 0]
    to80_mean = np.mean(to80) if to80 else float("nan")
    to80_rate = len(to80) / len(rows) * 100
    return {
        "method": method,
        "curve_mean": arr.mean(axis=0),
        "curve_se": arr.std(axis=0) / np.sqrt(len(arr)),
        "final": final,
        "step4": step4,
        "to80_mean_steps": to80_mean,
        "to80_rate_pct": to80_rate,
    }


summary = {m: summarize(m) for m in METHODS}

print("\n4-row comparison (mean over 30 HouseExpo maps, lidar=70px, max=20 steps)")
print("-" * 86)
print(f"{'Method':<24} {'Step-4 cov':>12} {'Final cov':>11} {'Steps to 80%':>16} {'80% reach %':>14}")
print("-" * 86)
for m in METHODS:
    s = summary[m]
    print(f"{LABELS[m]:<24} {s['step4']:>11.1f}%  {s['final']:>10.1f}%  "
          f"{s['to80_mean_steps']:>14.1f}   {s['to80_rate_pct']:>12.1f}%")
print("-" * 86)

baseline_step4 = summary["heuristic"]["step4"]
baseline_final = summary["heuristic"]["final"]
print("\nDeltas vs heuristic baseline (the conventional method):")
for m in METHODS:
    s = summary[m]
    print(f"  {LABELS[m]:<24} step-4 Δ {s['step4'] - baseline_step4:+5.2f}pp   "
          f"final Δ {s['final'] - baseline_final:+5.2f}pp")

table_path = OUT_DIR / "4baseline_table.txt"
with table_path.open("w") as fh:
    fh.write("Method                  Step-4 cov  Final cov  Steps to 80%  80% reach %\n")
    for m in METHODS:
        s = summary[m]
        fh.write(f"{LABELS[m]:<22}  {s['step4']:>9.1f}%  {s['final']:>8.1f}%  "
                 f"{s['to80_mean_steps']:>11.1f}  {s['to80_rate_pct']:>10.1f}%\n")

plt.rcParams.update({
    "figure.facecolor": "#0c1019",
    "axes.facecolor": "#0c1019",
    "axes.edgecolor": "#3a4258",
    "axes.labelcolor": "#dfe6f0",
    "xtick.color": "#dfe6f0",
    "ytick.color": "#dfe6f0",
    "text.color": "#dfe6f0",
    "axes.grid": False,
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
steps = np.arange(1, MAX_STEPS + 1)
for m in METHODS:
    s = summary[m]
    ax.plot(steps, s["curve_mean"], color=COLORS[m], linewidth=2.2,
            label=LABELS[m], marker="o", markersize=4)
    ax.fill_between(steps, s["curve_mean"] - s["curve_se"],
                    s["curve_mean"] + s["curve_se"], color=COLORS[m], alpha=0.18)
ax.set_xlabel("frontier decision step", fontsize=12)
ax.set_ylabel("mean coverage (%)", fontsize=12)
ax.set_title("4-baseline ablation on 30 HouseExpo maps\nmean ± 1 SE",
             fontsize=13, pad=12)
ax.set_xlim(1, MAX_STEPS)
ax.set_ylim(0, 100)
ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142", fontsize=11)
ax.axvline(4, color="#3a4258", linestyle="--", linewidth=1, alpha=0.7)
ax.text(4.2, 5, "step 4\n(early-budget regime)", color="#aab1c0", fontsize=9, alpha=0.9)
fig.tight_layout()
curves_path = OUT_DIR / "12_4baseline_curves.png"
fig.savefig(curves_path, dpi=120, facecolor="#0c1019")
plt.close(fig)
print(f"\nWrote {curves_path}")

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
x = np.arange(len(METHODS))
step4_vals = [summary[m]["step4"] for m in METHODS]
final_vals = [summary[m]["final"] for m in METHODS]
w = 0.36
ax.bar(x - w / 2, step4_vals, w, color="#6fb6ff", label="step-4 coverage",
       edgecolor="#1a2030", linewidth=0.5)
ax.bar(x + w / 2, final_vals, w, color="#c576ff", label="final coverage (20 steps)",
       edgecolor="#1a2030", linewidth=0.5)
for i, v in enumerate(step4_vals):
    ax.text(i - w / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10, color="#dfe6f0")
for i, v in enumerate(final_vals):
    ax.text(i + w / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10, color="#dfe6f0")
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=11)
ax.set_ylabel("coverage (%)", fontsize=12)
ax.set_title("4-baseline ablation: step-4 (early budget) vs final coverage",
             fontsize=13, pad=12)
ax.set_ylim(0, 105)
ax.legend(loc="lower left", facecolor="#161b26", edgecolor="#2a3142", fontsize=11)
fig.tight_layout()
bars_path = OUT_DIR / "13_4baseline_bars.png"
fig.savefig(bars_path, dpi=120, facecolor="#0c1019")
plt.close(fig)
print(f"Wrote {bars_path}")

summary_path = OUT_DIR.parent / "4baseline_summary.json"
out_json = {m: {k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in s.items()} for m, s in summary.items()}
out_json["_meta"] = {"n_maps": len(rows), "max_steps": MAX_STEPS, "lidar": 70}
summary_path.write_text(json.dumps(out_json, indent=2))
print(f"Wrote {summary_path}")
