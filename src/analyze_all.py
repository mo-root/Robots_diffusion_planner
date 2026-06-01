"""Unified analysis: consume every experiment JSON, produce slide-ready figures
and a single summary.json that the explainer embeds.

Inputs (any subset, missing files just skip):
  results/experiments/exp1_results.json                  -- original N=30
  results/experiments_n100/exp1_results.json             -- bigger N
  results/experiments_complexity/exp_results.json        -- per-map complexity
  results/experiments_synthetic/grid_of_rooms_results.json
  results/experiments_synthetic/warehouse_results.json
  results/experiments_warehouse_indomain/grid_of_rooms_results.json
  results/experiments_warehouse_indomain/warehouse_results.json
  results/experiments_realtime/realtime_results.json

Outputs:
  results/analysis/summary.json
  results/analysis/figures/*.png
"""

import os, sys, json, math, statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/analysis")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

COL_DIFF = "#c576ff"
COL_BASE = "#ffb86c"
COL_GOOD = "#7ee787"
COL_BAD = "#ff7b72"
plt.rcParams.update({
    "figure.facecolor": "#0c1019",
    "axes.facecolor": "#0c1019",
    "axes.edgecolor": "#3a4258",
    "axes.labelcolor": "#dfe6f0",
    "xtick.color": "#8d97a8",
    "ytick.color": "#8d97a8",
    "text.color": "#dfe6f0",
    "axes.grid": True,
    "grid.color": "#1a2030",
    "grid.linewidth": 0.5,
})


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def per_step_stats(records, diff_key="diff_cov", base_key="base_cov"):
    """records: list of dicts with diff_cov and base_cov arrays.
    Returns per-step mean, SE, win count for each step."""
    if not records:
        return None
    maxlen = max(len(r[diff_key]) for r in records)
    out = []
    for s in range(maxlen):
        d_vals, b_vals = [], []
        dw = bw = tie = 0
        for r in records:
            dv = r[diff_key][s] if s < len(r[diff_key]) else r[diff_key][-1]
            bv = r[base_key][s] if s < len(r[base_key]) else r[base_key][-1]
            d_vals.append(dv); b_vals.append(bv)
            if abs(dv - bv) < 0.005: tie += 1
            elif dv > bv: dw += 1
            else: bw += 1
        deltas = [d - b for d, b in zip(d_vals, b_vals)]
        n = len(records)
        out.append({
            "step": s + 1, "n": n,
            "diff_mean": float(np.mean(d_vals)),
            "base_mean": float(np.mean(b_vals)),
            "diff_se": float(np.std(d_vals, ddof=1) / math.sqrt(n)) if n > 1 else 0,
            "base_se": float(np.std(b_vals, ddof=1) / math.sqrt(n)) if n > 1 else 0,
            "mean_delta": float(np.mean(deltas)),
            "se_delta": float(np.std(deltas, ddof=1) / math.sqrt(n)) if n > 1 else 0,
            "diff_wins": dw, "base_wins": bw, "ties": tie,
        })
    return out


def plot_curves(stats_dict, fname, title=""):
    """stats_dict: { label: per_step_stats_list }. Plots mean ± 95% CI for each."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    for label, stats in stats_dict.items():
        if not stats: continue
        steps = [s["step"] for s in stats]
        dm = np.array([s["diff_mean"] for s in stats])
        dse = np.array([s["diff_se"] for s in stats])
        bm = np.array([s["base_mean"] for s in stats])
        bse = np.array([s["base_se"] for s in stats])
        ax.fill_between(steps, dm - 1.96*dse, dm + 1.96*dse, color=COL_DIFF, alpha=0.18)
        ax.fill_between(steps, bm - 1.96*bse, bm + 1.96*bse, color=COL_BASE, alpha=0.18)
        ax.plot(steps, dm, color=COL_DIFF, linewidth=2.4, label=f"diffusion ({label})")
        ax.plot(steps, bm, color=COL_BASE, linewidth=2.4, label=f"baseline ({label})", linestyle="--")
    ax.axhline(0.8, color="#3a4258", linestyle=":", linewidth=1)
    ax.axhline(0.9, color="#3a4258", linestyle=":", linewidth=1)
    ax.set_xlabel("frontier-decision step")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142")
    if title: ax.set_title(title, color="#dfe6f0")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_delta_bars(stats, fname, title=""):
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=110)
    steps = [s["step"] for s in stats]
    d = np.array([s["mean_delta"] for s in stats]) * 100
    se = np.array([s["se_delta"] for s in stats]) * 100
    colors = [COL_DIFF if v >= 0 else COL_BASE for v in d]
    ax.bar(steps, d, color=colors, width=0.7)
    ax.errorbar(steps, d, yerr=1.96*se, fmt="none", ecolor="#dfe6f0", capsize=3, linewidth=1.2)
    ax.axhline(0, color="#3a4258", linewidth=1.2)
    ax.set_xlabel("step budget")
    ax.set_ylabel("Δ coverage (%)  diffusion − baseline")
    if title: ax.set_title(title, color="#dfe6f0")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_win_stack(stats, fname, n_total, title=""):
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=110)
    steps = [s["step"] for s in stats]
    dw = np.array([s["diff_wins"] for s in stats])
    bw = np.array([s["base_wins"] for s in stats])
    tw = np.array([s["ties"] for s in stats])
    ax.bar(steps, dw, color=COL_DIFF, label="diffusion wins")
    ax.bar(steps, tw, bottom=dw, color="#3a4258", label="tie")
    ax.bar(steps, bw, bottom=dw+tw, color=COL_BASE, label="baseline wins")
    ax.set_xlabel("step budget")
    ax.set_ylabel(f"maps (of {n_total})")
    if title: ax.set_title(title, color="#dfe6f0")
    ax.legend(loc="upper right", facecolor="#161b26", edgecolor="#2a3142")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_complexity_scatter(records, fname):
    """For each map: complexity proxy (initial_frontiers) vs diffusion advantage at step 4."""
    if not records: return
    xs, ys, colors = [], [], []
    for r in records:
        if "initial_frontiers" not in r: continue
        if len(r["diff_cov"]) < 4 or len(r["base_cov"]) < 4: continue
        d4 = r["diff_cov"][3]; b4 = r["base_cov"][3]
        xs.append(r["initial_frontiers"])
        ys.append((d4 - b4) * 100)
        colors.append(COL_DIFF if d4 > b4 else (COL_BASE if d4 < b4 else "#8d97a8"))
    if not xs: return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    ax.scatter(xs, ys, c=colors, s=60, alpha=0.85, edgecolors="#0c1019")
    ax.axhline(0, color="#3a4258", linewidth=1)
    # trend line
    xs_a = np.array(xs); ys_a = np.array(ys)
    if len(xs_a) > 2:
        coef = np.polyfit(xs_a, ys_a, 1)
        xr = np.array([xs_a.min(), xs_a.max()])
        ax.plot(xr, np.polyval(coef, xr), color=COL_GOOD, linewidth=2, linestyle="--",
                label=f"slope = {coef[0]:+.2f}%/frontier")
    ax.set_xlabel("initial frontier count (complexity proxy)")
    ax.set_ylabel("Δ coverage at step 4 (%)")
    ax.set_title("Does diffusion advantage scale with map complexity?", color="#dfe6f0")
    ax.legend(facecolor="#161b26", edgecolor="#2a3142")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_realtime_compare(records, fname):
    """3-way: baseline, discrete-diffusion, realtime-diffusion."""
    if not records: return
    maxlen = max(len(r["base_cov"]) for r in records)
    def aggr(key):
        out = []
        for s in range(maxlen):
            vals = [r[key][s] if s < len(r[key]) else r[key][-1] for r in records]
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0
            out.append((m, se))
        return out
    base = aggr("base_cov")
    disc = aggr("disc_cov")
    rt = aggr("rt_cov")
    steps = list(range(1, maxlen + 1))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    for series, color, label in [(base, COL_BASE, "baseline"),
                                  (disc, COL_DIFF, "diffusion (per-decision)"),
                                  (rt, COL_GOOD, "diffusion (realtime re-plan)")]:
        m = np.array([v[0] for v in series])
        se = np.array([v[1] for v in series])
        ax.fill_between(steps, m - 1.96*se, m + 1.96*se, color=color, alpha=0.16)
        ax.plot(steps, m, color=color, linewidth=2.4, label=label)
    ax.set_xlabel("frontier-decision step")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142")
    ax.set_title("Realtime re-planning vs per-decision sampling", color="#dfe6f0")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_in_vs_ood(in_dom, ood, fname):
    """Side-by-side per-step delta: in-domain vs OOD. The killer figure."""
    fig, ax = plt.subplots(figsize=(9, 4), dpi=110)
    steps = [s["step"] for s in in_dom]
    n = min(len(in_dom), len(ood))
    in_d = np.array([s["mean_delta"] for s in in_dom[:n]]) * 100
    in_se = np.array([s["se_delta"] for s in in_dom[:n]]) * 100
    ood_d = np.array([s["mean_delta"] for s in ood[:n]]) * 100
    ood_se = np.array([s["se_delta"] for s in ood[:n]]) * 100
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w/2, in_d, w, color=COL_DIFF, label="HouseExpo-trained (in-domain)", alpha=0.95)
    ax.bar(x + w/2, ood_d, w, color="#5a6480", label="warehouse-trained (out-of-domain)", alpha=0.95)
    ax.errorbar(x - w/2, in_d, yerr=1.96*in_se, fmt="none", ecolor="#dfe6f0", capsize=2.5, linewidth=1)
    ax.errorbar(x + w/2, ood_d, yerr=1.96*ood_se, fmt="none", ecolor="#dfe6f0", capsize=2.5, linewidth=1)
    ax.axhline(0, color="#3a4258", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in steps[:n]])
    ax.set_xlabel("step budget")
    ax.set_ylabel("Δ coverage vs baseline (%)")
    ax.set_title("Same maps, same scoring, only the prior differs", color="#dfe6f0")
    ax.legend(loc="upper right", facecolor="#161b26", edgecolor="#2a3142")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def plot_in_vs_ood_winrate(orig, cross, fname):
    """Per-step: wins vs losses for in-domain vs OOD."""
    n = min(max(len(r["diff_cov"]) for r in orig), max(len(r["diff_cov"]) for r in cross))
    in_w = []; in_l = []
    ood_w = []; ood_l = []
    for s in range(n):
        iw = il = 0
        for r in orig:
            dv = r["diff_cov"][s] if s < len(r["diff_cov"]) else r["diff_cov"][-1]
            bv = r["base_cov"][s] if s < len(r["base_cov"]) else r["base_cov"][-1]
            if dv - bv > 0.005: iw += 1
            elif bv - dv > 0.005: il += 1
        in_w.append(iw); in_l.append(il)
        ow = ol = 0
        for r in cross:
            dv = r["diff_cov"][s] if s < len(r["diff_cov"]) else r["diff_cov"][-1]
            bv = r["base_cov"][s] if s < len(r["base_cov"]) else r["base_cov"][-1]
            if dv - bv > 0.005: ow += 1
            elif bv - dv > 0.005: ol += 1
        ood_w.append(ow); ood_l.append(ol)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=110, sharey=True)
    x = np.arange(n)
    for ax, w, l, title, color in [(axes[0], in_w, in_l, "in-domain (HouseExpo prior)", COL_DIFF),
                                    (axes[1], ood_w, ood_l, "out-of-domain (warehouse prior)", "#5a6480")]:
        ax.bar(x, w, color=color, label="diffusion wins")
        ax.bar(x, [-v for v in l], color=COL_BASE, label="baseline wins")
        ax.axhline(0, color="#3a4258", linewidth=1)
        ax.set_xlabel("step budget")
        ax.set_title(title, color="#dfe6f0")
        ax.set_xticks(x); ax.set_xticklabels([str(i+1) for i in range(n)])
        if ax is axes[0]:
            ax.set_ylabel("map count (positive = diff wins)")
        ax.legend(loc="lower right", facecolor="#161b26", edgecolor="#2a3142", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=120, facecolor="#0c1019")
    plt.close(fig)
    print(f"  wrote {fname}")


def summarize_set(records, name):
    """Return a dict of headline stats for one experiment set."""
    if not records: return None
    stats = per_step_stats(records)
    if not stats: return None
    return {
        "name": name,
        "n_maps": len(records),
        "per_step": stats,
        "peak_step": max(range(len(stats)), key=lambda i: stats[i]["mean_delta"]) + 1,
        "peak_delta": max(s["mean_delta"] for s in stats),
        "step4_delta": stats[3]["mean_delta"] if len(stats) > 3 else None,
        "step4_wins": (stats[3]["diff_wins"], stats[3]["base_wins"], stats[3]["ties"]) if len(stats) > 3 else None,
        "final_delta": stats[-1]["mean_delta"],
        "mean_steps_to_80": {
            "diff": float(np.mean([r.get("diff_to_80", -1) for r in records if r.get("diff_to_80", -1) > 0])),
            "base": float(np.mean([r.get("base_to_80", -1) for r in records if r.get("base_to_80", -1) > 0])),
        },
    }


def main():
    summary = {}
    print("Loading experiments...")

    # 1. Original N=30
    orig = load_json("results/experiments/exp1_results.json")
    if orig:
        summary["houseexpo_n30"] = summarize_set(orig, "HouseExpo N=30")
        plot_delta_bars(per_step_stats(orig), "01_houseexpo_n30_delta.png",
                       "HouseExpo N=30: per-step coverage advantage")
        plot_win_stack(per_step_stats(orig), "02_houseexpo_n30_wins.png", len(orig),
                      "HouseExpo N=30: per-step win breakdown")

    # 2. N=100
    n100 = load_json("results/experiments_n100/exp1_results.json")
    if n100:
        summary["houseexpo_n100"] = summarize_set(n100, "HouseExpo N=100")

    # 2b. Cross-domain: warehouse-trained model on HouseExpo
    cross = load_json("results/experiments_warehouse_on_houseexpo/exp1_results.json")
    if cross:
        summary["warehouse_on_houseexpo"] = summarize_set(cross, "Warehouse model on HouseExpo (OOD)")
        # Plot directly comparable: original (in-domain) vs cross (OOD), same metric
        if orig:
            in_dom_stats = per_step_stats(orig)
            ood_stats = per_step_stats(cross)
            plot_in_vs_ood(in_dom_stats, ood_stats, "07_in_vs_ood_delta.png")
            plot_in_vs_ood_winrate(orig, cross, "08_in_vs_ood_wins.png")

    # 3. Complexity stratification
    cx = load_json("results/experiments_complexity/exp_results.json")
    if cx:
        summary["complexity"] = summarize_set(cx, "Complexity-stratified N=80")
        plot_complexity_scatter(cx, "03_complexity_scatter.png")

    # 4. Synthetic (HouseExpo-trained model)
    syn_grid = load_json("results/experiments_synthetic/grid_of_rooms_results.json")
    syn_wh = load_json("results/experiments_synthetic/warehouse_results.json")
    if syn_grid or syn_wh:
        sets = {}
        if syn_grid: sets["grid-of-rooms"] = per_step_stats(syn_grid)
        if syn_wh: sets["warehouse"] = per_step_stats(syn_wh)
        plot_curves(sets, "04_synthetic_curves.png",
                   "Synthetic envs with HouseExpo-trained model (OOD)")
        if syn_grid: summary["synthetic_grid_houseexpo_model"] = summarize_set(syn_grid, "grid_of_rooms (HouseExpo model)")
        if syn_wh: summary["synthetic_warehouse_houseexpo_model"] = summarize_set(syn_wh, "warehouse (HouseExpo model)")

    # 5. Synthetic (warehouse-trained model — in-domain)
    wd_grid = load_json("results/experiments_warehouse_indomain/grid_of_rooms_results.json")
    wd_wh = load_json("results/experiments_warehouse_indomain/warehouse_results.json")
    if wd_grid or wd_wh:
        sets = {}
        if wd_grid: sets["grid-of-rooms"] = per_step_stats(wd_grid)
        if wd_wh: sets["warehouse"] = per_step_stats(wd_wh)
        plot_curves(sets, "05_indomain_curves.png",
                   "Synthetic envs with warehouse-trained model (in-domain)")
        if wd_grid: summary["synthetic_grid_warehouse_model"] = summarize_set(wd_grid, "grid_of_rooms (warehouse model)")
        if wd_wh: summary["synthetic_warehouse_warehouse_model"] = summarize_set(wd_wh, "warehouse (warehouse model)")

    # 6. Realtime
    rt = load_json("results/experiments_realtime/realtime_results.json")
    if rt:
        plot_realtime_compare(rt, "06_realtime_compare.png")
        # for summary, compare rt_cov vs base_cov and rt_cov vs disc_cov
        n = len(rt)
        maxlen = max(len(r["base_cov"]) for r in rt)
        def m(key, s):
            vals = [r[key][s] if s < len(r[key]) else r[key][-1] for r in rt]
            return float(np.mean(vals))
        summary["realtime"] = {
            "name": "Realtime continuous re-sampling",
            "n_maps": n,
            "rt_step4": m("rt_cov", 3),
            "disc_step4": m("disc_cov", 3),
            "base_step4": m("base_cov", 3),
            "rt_step7": m("rt_cov", 6) if maxlen > 6 else None,
            "disc_step7": m("disc_cov", 6) if maxlen > 6 else None,
            "base_step7": m("base_cov", 6) if maxlen > 6 else None,
            "rt_final": m("rt_cov", maxlen - 1),
            "disc_final": m("disc_cov", maxlen - 1),
            "base_final": m("base_cov", maxlen - 1),
        }

    # 7. Cross-domain matrix
    summary["domain_matrix"] = build_domain_matrix(summary)

    # write summary
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT/'summary.json'}")
    print(f"Wrote figures to {FIG}/")
    print_headline(summary)


def build_domain_matrix(summary):
    """Construct a small (model x test-env) matrix of step-4 delta."""
    rows = []
    pairs = [
        ("HouseExpo model", "HouseExpo test", "houseexpo_n30"),
        ("warehouse model", "HouseExpo test", "warehouse_on_houseexpo"),
        ("HouseExpo model", "synthetic grid", "synthetic_grid_houseexpo_model"),
        ("HouseExpo model", "synthetic warehouse", "synthetic_warehouse_houseexpo_model"),
        ("warehouse model", "synthetic grid", "synthetic_grid_warehouse_model"),
        ("warehouse model", "synthetic warehouse", "synthetic_warehouse_warehouse_model"),
    ]
    for model, env, key in pairs:
        s = summary.get(key)
        if not s: continue
        rows.append({"model": model, "test_env": env,
                     "step4_delta_pct": s["step4_delta"] * 100 if s["step4_delta"] is not None else None,
                     "step4_wins": s.get("step4_wins"),
                     "peak_step": s.get("peak_step"),
                     "peak_delta_pct": s["peak_delta"] * 100 if s["peak_delta"] is not None else None})
    return rows


def print_headline(summary):
    print("\n" + "="*60)
    print("HEADLINE NUMBERS")
    print("="*60)
    for key in ["houseexpo_n30", "houseexpo_n100", "warehouse_on_houseexpo", "complexity",
                "synthetic_grid_houseexpo_model", "synthetic_warehouse_houseexpo_model",
                "synthetic_grid_warehouse_model", "synthetic_warehouse_warehouse_model"]:
        s = summary.get(key)
        if not s: continue
        delta = s["step4_delta"] * 100 if s["step4_delta"] is not None else 0
        wins = s.get("step4_wins")
        print(f"\n{s['name']} (n={s['n_maps']}):")
        print(f"  step-4 Δ = {delta:+.2f}%   wins: {wins}")
        print(f"  peak step {s['peak_step']}: Δ = {s['peak_delta']*100:+.2f}%")
        print(f"  final Δ = {s['final_delta']*100:+.2f}%")
    rt = summary.get("realtime")
    if rt:
        print(f"\nRealtime (n={rt['n_maps']}):")
        print(f"  step 4: rt={rt['rt_step4']:.3f}  disc={rt['disc_step4']:.3f}  base={rt['base_step4']:.3f}")
        if rt['rt_step7'] is not None:
            print(f"  step 7: rt={rt['rt_step7']:.3f}  disc={rt['disc_step7']:.3f}  base={rt['base_step7']:.3f}")
        print(f"  final : rt={rt['rt_final']:.3f}  disc={rt['disc_final']:.3f}  base={rt['base_final']:.3f}")
    print("\nDomain matrix:")
    for row in summary.get("domain_matrix", []):
        delta = row["step4_delta_pct"]
        if delta is None: continue
        wins = row.get("step4_wins") or ("?", "?", "?")
        print(f"  {row['model']:>20s}  →  {row['test_env']:>22s}  Δ@4 = {delta:+.2f}%   wins: {wins}")


if __name__ == "__main__":
    main()
