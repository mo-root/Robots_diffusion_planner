# Failure Analysis — Diffusion-Guided Frontier Exploration (exp1, N=30)

Source: `results/experiments/exp1_results.json` · Figures: `09_failure_cases.png`, `10_win_examples.png`

## Hard failures (diff_to_80 == -1)

Three maps where the diffusion-guided planner never crossed 80% coverage in the 20-step budget.

| map_idx | diff final | base final | base_to_80 | failure mode |
|---|---|---|---|---|
| 4363  | 0.77 | 0.80 | 7  | **plateau just below 80%** |
| 10795 | 0.36 | 1.00 | 6  | **early collapse** |
| 17925 | 0.27 | 0.30 | -1 | **both stall** (likely disconnected map) |

Trajectory notes:
- **4363** tracks baseline through step 3 (~0.55), then crawls in ~1% increments while baseline accelerates to 80% by step 7. A scoring tie broken the wrong way by the imagined completions.
- **10795** diverges immediately: by step 3 diff=0.35 vs base=0.70; diffusion adds essentially 0% from step 6 onward. Stuck-against-an-imagined-wall.
- **17925** is the surprise: **baseline also fails** (0.30 final). Both stall by step 3 — almost certainly an unreachable-region map issue, not a diffusion failure.

Strictly under (diff fails AND base succeeds), only **2 of 30** are diffusion-specific hard failures (4363, 10795).

## Large losses (>3 steps slower than baseline, both reach 80%)

Only one map qualifies:

| map_idx | diff_to_80 | base_to_80 | delta | diff final | base final |
|---|---|---|---|---|---|
| 19006 | 13 | 7 | +6 | 0.98 | 0.91 |

Diffusion is far slower to 80% on 19006 but ultimately covers **more** of the map (0.98 vs 0.91). Baseline hits 80% fast then stalls; diffusion grinds early, breaks out around step 15.

## Biggest wins (≥2 steps faster, ranked by absolute delta)

| map_idx | diff_to_80 | base_to_80 | speedup |
|---|---|---|---|
| 17666 | 5 | 9 | **−4 steps** |
| 1437  | 5 | 7 | −2 steps |
| 6147  | 6 | 8 | −2 steps |

On 17666 the diffusion curve jumps to 0.86 by step 4 while the baseline is still at 0.58, and it crosses 80% nearly twice as fast.

## Qualitative pattern: wins vs losses

Looking only at the trajectories:

1. **Wins open the gap early (steps 2-4).** In all three win panels, diffusion is meaningfully ahead by step 3 — the imagined-completion score steers the robot into a high-yield wing before baseline commits. The lead is preserved, not extended; curves run parallel past 80%.
2. **Losses are early plateaus, not slow starts.** 4363 and 10795 share baseline's first 2 steps, then flatten. The planner is not making worse greedy choices; it makes *no progress at all* for 5-15 steps — picking frontiers the model hallucinates as informative but that turn out to be walls.
3. **No overshoot.** No run shows diffusion racing past baseline then crashing. Failure is always silence, never regression.
4. **Final coverage decouples from steps-to-80%.** Map 19006: 6 steps slower to 80% yet 7 points higher at the end. The diffusion prior favors thoroughness over speed on harder layouts.

## Take-away for the slides

- The "3 hard failures" are really **2 + 1**: two diffusion-specific stalls (4363, 10795) and one map (17925) that defeats both methods.
- The diffusion model wins by **opening a gap early** when its imagined completions are correct, and loses by **getting stuck on hallucinated frontiers** when they are not — failure is silence, not error.

---

Moin Mattar

AI helped me in formatting and writing (LaTeX/Markdown), as well as explained concepts.
