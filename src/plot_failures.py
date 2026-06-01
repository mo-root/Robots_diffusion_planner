"""Generate failure-case and big-win figures for the exp1 HouseExpo N=30 results.

Produces two PNGs in results/analysis/figures/:
  - 09_failure_cases.png : 3 panels for the 3 hard-failure maps
  - 10_win_examples.png  : 3 panels for the 3 biggest wins

Run from project root:
    python -m src.plot_failures
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- styling -------------------------------------------------------------- #

BG = "#0c1019"
FG = "#e6e8ee"
GRID = "#2a3144"
DIFF_COLOR = "#c576ff"
BASE_COLOR = "#ffb86c"
REF_COLOR = "#54607a"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = PROJECT_ROOT / "results" / "experiments" / "exp1_results.json"
OUT_DIR = PROJECT_ROOT / "results" / "analysis" / "figures"


def _load_runs() -> list[dict]:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _classify(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (failures, wins) — each a list of 3 run dicts in plot order."""
    # Failures: any run where diff_to_80 == -1.  There are exactly three.
    failures = [r for r in runs if r["diff_to_80"] == -1]
    failures.sort(key=lambda r: r["diff_final"])  # worst → least-bad

    # Wins: positive (base_to_80 - diff_to_80), break ties by ratio.
    candidates = [
        r
        for r in runs
        if r["diff_to_80"] != -1
        and r["base_to_80"] != -1
        and (r["base_to_80"] - r["diff_to_80"]) >= 1
    ]
    candidates.sort(
        key=lambda r: (
            r["base_to_80"] - r["diff_to_80"],
            r["base_to_80"] / max(r["diff_to_80"], 1),
        ),
        reverse=True,
    )
    wins = candidates[:3]
    return failures, wins


def _failure_mode(run: dict) -> str:
    if run["base_to_80"] == -1:
        return "both stall"
    if run["diff_final"] < 0.5:
        return "early collapse"
    return "plateau below 80%"


def _style_axis(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=12)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Step", color=FG, fontsize=14)
    ax.set_ylabel("Coverage", color=FG, fontsize=14)
    ax.set_ylim(0.0, 1.02)


def _plot_run(ax, run: dict, title: str) -> None:
    diff = run["diff_cov"]
    base = run["base_cov"]
    ax.plot(
        range(len(diff)),
        diff,
        color=DIFF_COLOR,
        linewidth=2.6,
        marker="o",
        markersize=5,
        label="diffusion",
    )
    ax.plot(
        range(len(base)),
        base,
        color=BASE_COLOR,
        linewidth=2.6,
        marker="s",
        markersize=5,
        label="baseline",
    )
    ax.axhline(0.80, color=REF_COLOR, linestyle="--", linewidth=1.2, alpha=0.85)
    ax.axhline(0.90, color=REF_COLOR, linestyle=":", linewidth=1.2, alpha=0.85)
    ax.set_title(title, color=FG, fontsize=15, pad=10)
    _style_axis(ax)


def _make_figure(runs: list[dict], titles: list[str], out_path: Path, suptitle: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    for ax, run, title in zip(axes, runs, titles):
        _plot_run(ax, run, title)
    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color=REF_COLOR, linestyle="--", linewidth=1.2))
    labels.append("80% / 90% targets")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        facecolor=BG,
        edgecolor=GRID,
        labelcolor=FG,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(suptitle, color=FG, fontsize=17, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = _load_runs()
    failures, wins = _classify(runs)
    assert len(failures) == 3, f"expected 3 failure maps, got {len(failures)}"
    assert len(wins) == 3, f"expected 3 win maps, got {len(wins)}"

    fail_titles = []
    for r in failures:
        base_note = (
            f"base→80% @ step {r['base_to_80']}"
            if r["base_to_80"] != -1
            else f"base also stalls ({r['base_final']:.2f})"
        )
        fail_titles.append(
            f"map {r['map_idx']} — {_failure_mode(r)}\n"
            f"diff final {r['diff_final']:.2f} · {base_note}"
        )
    _make_figure(
        failures,
        fail_titles,
        OUT_DIR / "09_failure_cases.png",
        "Failure cases — diffusion never reaches 80% coverage",
    )

    win_titles = [
        f"map {r['map_idx']} — diff→80% @ {r['diff_to_80']}, base@ {r['base_to_80']}\n"
        f"speedup {r['base_to_80'] - r['diff_to_80']} steps"
        for r in wins
    ]
    _make_figure(
        wins,
        win_titles,
        OUT_DIR / "10_win_examples.png",
        "Biggest wins — diffusion reaches 80% well before baseline",
    )

    print("wrote:", OUT_DIR / "09_failure_cases.png")
    print("wrote:", OUT_DIR / "10_win_examples.png")
    print("failures:", [r["map_idx"] for r in failures])
    print("wins:", [r["map_idx"] for r in wins])


if __name__ == "__main__":
    main()
