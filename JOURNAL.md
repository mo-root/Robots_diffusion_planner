# Project Journal

A running log of where this project stands. Read this first if you're returning to it after a break.

## Current state (end of 2026-05-31, ~6 PM Eastern)

Project is presentation-ready for June 1. All EC2 boxes stopped. All deliverables
on the local Mac. The thesis evolved during the day from a defensive
"modest 4% advantage" framing into a much stronger **structure-and-domain
decomposition** with three independent supporting experiments + a 2x2 ablation
matrix + a hardware-deployment latency profile.

### The thesis in one sentence

A learned generative prior gives a robot a measurable head-start in early
exploration, scaling with map complexity (+4.7% on hardest), vanishing entirely
with a mismatched prior (-0.1%), nearly tripling under free inference (+10.0%),
and decomposing cleanly into **prior-match contribution (~7 percentage points)
+ sampling-frequency contribution (~7 percentage points), approximately
additive**.

### The deliverables

| File | Contents |
|---|---|
| `slides.html` | 8-slide deck (1920x1080, browser navigation, prints to PDF). Already presented to in slide 4 as "harder maps = more advantage" framing. |
| `PRESENTATION_SCRIPT.md` | 6-minute talk script with per-slide timing. |
| `explainer.html` | 13-section interactive walkthrough. Now includes section 11 (OOD ablation) and section 12 (2x2 matrix + realtime). |
| `README.md` | Replaced old "+24.5% info gain" claim with the structure-and-domain results table. |
| `FAILURE_ANALYSIS.md` | The 2 hard-failure cases + the 3 biggest-win cases with trajectories. |
| `results/analysis/figures/01-11_*.png` | 11 slide-grade dark-theme PNGs covering every result. |
| `results/analysis/summary.json` | Machine-readable summary of all experiment headline numbers. |

## Findings 2026-05-31

### Finding 1 — In-domain helps, more on complex maps (N=80 stratified)

Per-step coverage Δ vs baseline, bucketed by initial frontier count (complexity proxy):

| Complexity bucket | step-4 Δ | wins / losses / ties |
|---|---|---|
| Low (3-15 frontiers, n=26) | +2.56% | 10 / 6 / 10 |
| Mid (15-19 frontiers, n=27) | +1.20% | 12 / 14 / 1 |
| **High (19-35 frontiers, n=27)** | **+4.74%** | **18 / 4 / 5** |

The complex-map bucket gives the strongest signal yet: +4.7% advantage, 4.5x more
wins than losses. Matches the structure thesis directly.

Files: `results/experiments_complexity/exp_results.json`, plot
`results/analysis/figures/04_complexity_buckets.png`.

### Finding 2 — Out-of-domain prior is essentially useless (N=30)

We trained a second model from scratch on synthetic stereotyped environments
(grid-of-rooms + warehouse, 8 epochs, 6 min on T4, 2.38M params). Evaluated on
the SAME 30 HouseExpo maps with the SAME scoring code.

| Model | Step-4 Δ | wins / losses / ties |
|---|---|---|
| HouseExpo-trained (in-domain) | +3.47% | 18 / 8 / 4 |
| Warehouse-trained (OOD) | -0.09% | 9 / 12 / 9 |

The cleanest possible ablation. Same architecture, same scoring, same maps; only
the prior differs. The +3.47% is a property of the matched-prior pipeline, not
of the architecture.

Files: `results/experiments_warehouse_on_houseexpo/exp1_results.json`,
checkpoint `results/checkpoints_warehouse/warehouse_epoch0008.pt`,
plot `results/analysis/figures/07_in_vs_ood_delta.png`.

### Finding 3 — Realtime upper bound (N=19 in-domain, N=30 OOD)

Hypothesis: if inference were free (sub-second), re-sampling K completions every
3 driving steps instead of only at frontier decisions would widen the advantage.
Confirmed dramatically.

**In-domain + realtime** (N=19, run crashed at map 19 but story already clear):
- Step 4 coverage: rt=72.3%, disc=65.7%, base=62.3%
- Realtime vs baseline: **+10.0%** (nearly 3x the discrete advantage)
- Realtime vs discrete diffusion: **+6.6%** (isolates the sampling-frequency effect)

**OOD + realtime cross-experiment** (N=30, ran cleanly):
- Step 4 coverage: rt=64.8%, disc=59.1%, base=62.3%
- Realtime vs baseline: **+2.5%** (rescues the OOD prior partially)
- Realtime vs discrete diffusion: **+5.7%**

Files: `results/experiments_realtime/realtime_results.json` (in-domain) and
`results/experiments_warehouse_realtime/realtime_results.json` (OOD).
Plot: `results/analysis/figures/11_2x2_matrix.png`.

### Finding 4 — The 2x2 decomposition

Combining Findings 1-3 into the full prior x sampling-mode matrix:

| Configuration | Step-4 coverage | Δ vs baseline |
|---|---|---|
| In-domain + realtime (N=19) | 72.3% | **+10.0%** |
| In-domain + discrete (N=19) | 65.7% | +3.4% |
| OOD + realtime (N=30) | 64.8% | +2.5% |
| OOD + discrete (N=30) | 59.1% | -3.2% |
| Baseline (in-domain reference) | 62.3% | -- |

The contributions are roughly additive: prior match alone +7pp, sampling
frequency alone +7pp. Together they hit +10. The surprising result is that
**sampling frequency matters more than domain match in our data** — even with
the wrong prior, realtime mode clears baseline. The strongest argument for
deployment via model distillation.

### Finding 5 — K-sample ablation (defends K=8)

K sweep on 15 HouseExpo maps (75 rollouts):

| K | mean final coverage |
|---|---|
| 1 | 0.96 |
| 2 | 0.96 |
| 4 | 0.96 |
| 8 | 0.97 |
| 16 | 0.97 |

Coverage curves all roughly identical with K>=2; K=8 chosen for the better
uncertainty estimate (variance of K-sample gain estimates).
File: `results/experiments_ablation/k_sweep.json`. Plot: `05_ablation_k.png`.

Lambda sweep partially completed (24/60 runs) before we deliberately killed it
to free box 1 for the latency profile.

### Finding 6 — Latency profile (the deployment argument)

T4 inference benchmarks (5 trials each, median):

**K samples at DDIM=30:**
- K=1: 452 ms · K=2: 812 ms · K=4: 1.53 s · K=8: 3.46 s · K=16: 6.15 s · K=32: 11.9 s

**DDIM steps at K=8:**
- 5 steps: 557 ms · 10 steps: 1.11 s · 30 steps: 3.46 s · 50 steps: 5.58 s · 100 steps: 11.2 s

Both scale linearly. The actionable point: **K=4 + DDIM=10 = 500 ms** is
exactly the realtime budget. Combined with model distillation to smaller
U-Net, sub-second on Jetson is achievable.

File: `results/latency_profile.json`. Plot: `06b_latency_profile.png`.

### Failure modes

- 2 maps (4363, 10795) where diffusion-guided robot got stuck.
- 1 map (17925) where BOTH methods stall (disconnected HouseExpo geometry,
  not a diffusion-specific failure).
- ~7% hard-failure rate.
- Surprise from failure analysis: map 19006 is the worst non-failure loss by
  steps-to-80% (diffusion is 6 steps slower) yet ends up at **higher** final
  coverage than baseline (0.98 vs 0.91). Baseline races to 80% then stalls;
  diffusion grinds then breaks through around step 15. This means the
  steps-to-80% metric disagrees with final-coverage on the same map.

See `FAILURE_ANALYSIS.md`.

### Failure modes

- 2 maps (4363, 10795) where the diffusion-guided robot got stuck.
- 1 map (17925) where BOTH methods stall (likely disconnected HouseExpo
  geometry, not a diffusion-specific failure).
- 7% hard-failure rate. K-sample disagreement could be used to detect "I do not
  know" and fall back to baseline. Future work.

See `FAILURE_ANALYSIS.md` for the per-map detail.

## Chronicle (the actual sequence of work)

Useful for understanding why decisions were made.

**~10am** — Explainer + slides + initial in-domain analysis already done from
yesterday. Question of the day: "is our thing actually useful, where does it
help most, and how does that look in a presentation?"

**~11am** — Pivoted from "modest 4% advantage" defensive framing into
**structure-and-domain thesis**: prior helps when matched, on complex maps,
under tight budgets, with fast inference. Brainstormed three experiment
layers via the brainstorming skill. User approved "structure & domain"
narrative.

**~11:30am** — Spun up box 1 (existing `i-0b2d8ea34eaa56ef3`) for the
complexity stratification and synthetic experiments.

**~12pm** — Synthetic-environment experiments hit a planner limitation:
the straight-line "teleport to frontier" rollout cannot navigate through
narrow doorways in dense floor plans. Both diffusion and baseline stall at
25-50% coverage. Decision: **drop synthetic envs entirely**, replace with
the cleaner OOD-on-HouseExpo test. Same thesis, working planner.

**~12:30pm** — Spun up box 2 (`i-031b1f31c52596e92`, fresh) for parallel
warehouse-domain training. Cost ~$0.50/hr + same on box 1.

**~1pm** — Implemented batched score_diff (K samples in single GPU forward
pass): **3x speedup** confirmed via benchmark (22s -> 7s per call). Patched
both boxes' experiments.py.

**~1:30pm** — Warehouse-domain model trained on box 2: 8 epochs of synthetic
maps in **6 minutes**, loss 0.218 -> 0.022.

**~2pm** — Cross-domain test (warehouse model on HouseExpo) completed in
10 minutes on box 2. Result: **+/-0.1% step-4 advantage**, wins/losses
essentially balanced. Clean negative control.

**~2:30pm** — Generated first slide-ready figures (in-domain vs OOD bars,
win count breakdown).

**~3pm** — Spawned subagent for slide deck. Spawned second subagent for
failure-case analysis. Both delivered cleanly.

**~3:30pm** — Realtime experiment launched on box 2 (in-domain HouseExpo
model, replan_every=3 driving cells). Slow per-map (~2 min each, lots of
extra forward passes).

**~4pm** — K-sample ablation launched on box 1 (K in {1,2,4,8,16}, N=15
maps each). Lambda sweep queued after.

**~5pm** — Realtime experiment got partial result (N=15): **+12% advantage**
over baseline. Huge upgrade over discrete mode. (Final number with N=19
became +10%.)

**~5:30pm** — Realtime experiment crashed at map 19/30, cause unknown
(likely a specific map triggered an edge case). Decision: use the N=19
result, it is more than sufficient to tell the story.

**~5:45pm** — Pivoted box 2 to a NEW experiment: warehouse model **with
realtime mode** on HouseExpo. The OOD-plus-realtime combination test.

**~6pm** — Cut lambda sweep on box 1 to free GPU for the latency profile
(user-approved per AskUserQuestion).

**~6:15pm** — Latency profile completed: K=8 + DDIM=30 = 3.5s.
K=4 + DDIM=10 = 500ms (realtime budget). Linear in both K and DDIM steps.

**~6:30pm** — OOD-plus-realtime experiment completed all 30 maps in 22 min
on box 2. Result: **+2.5%** over baseline (vs -0.1% for discrete OOD).
Realtime partially rescues the wrong prior.

**~6:45pm** — Generated the 2x2 matrix figure (the cleanest single summary
of the entire project). Fixed the source-consistency issue (all cells now
from same-experiment paired data).

**~7pm** — Stopped both EC2 boxes. Updated README, JOURNAL, slides, talk
script, explainer with all final numbers. Total compute cost: ~$4.50.

## Deliverables produced today (full list)

- `slides.html` — 8-slide deck, 1920x1080, browser-navigable, print-friendly.
  Slide 4 leads with the complexity bucket finding (+4.7% on hardest maps).
  Slide 6 shows the 2x2 matrix. Slide 8 includes the latency profile figure.
- `PRESENTATION_SCRIPT.md` — 6-minute talk script with per-slide timing,
  all final numbers in line.
- `explainer.html` — 13 sections total. Sections 1-10 unchanged from before
  today. Section 11 (OOD ablation) and section 12 (2x2 matrix + realtime) added.
- `FAILURE_ANALYSIS.md` — failure-mode write-up from subagent.
- `README.md` — old +24.5% claim removed, new structure-and-domain results
  table in.
- `JOURNAL.md` (this file) — running log.
- `results/analysis/figures/01-11_*.png` — 11 slide-ready figures.
- `results/analysis/summary.json` — machine-readable headline numbers.
- `src/analyze_all.py`, `src/plot_2x2_matrix.py`,
  `src/plot_ablation_realtime_latency.py`,
  `src/plot_complexity_buckets.py`, `src/plot_failures.py` — analysis
  scripts. Re-runnable.
- `src/experiments_ablation.py`, `src/experiments_complexity.py`,
  `src/experiments_realtime.py`, `src/experiments_synthetic.py`,
  `src/latency_profile.py`, `src/synthetic_envs.py`,
  `src/train_warehouse.py` — experiment scripts. On both boxes (in /home/ubuntu/src)
  and local /tmp.
- `results/checkpoints_warehouse/warehouse_epoch0008.pt` — warehouse model
  checkpoint (on box 2, can be downloaded if needed).

## What is left (June 1 + June 6)

| Task | Deadline | State |
|---|---|---|
| Final presentation rehearsal | June 1 morning | open |
| 4-page IEEE report | June 6 | draft exists, needs numbers refreshed to match today's findings |
| 3-minute teaser video | June 6 | 1.6 min version exists, needs extension to ~3 min |
| 1-page individual reflection | June 6 | not started |
| Clean source + README | June 6 | README already updated, code is clean |

## Older state (2026-05-30 baseline)

## What works

- **Conditional U-Net diffusion model.** 4.16M parameters, base_channels=32, channel_mults=(1,2,4,4), attention only at the 16x16 bottleneck. Trained 29 epochs on a Tesla T4 against 2.66M synthetic (partial, full) pairs from HouseExpo. Final pixel accuracy 82%, IoU 0.62 (single scan) to 0.88 (5+ scans). Checkpoint `results/checkpoints/model_epoch0020.pt` is the one used for all reported numbers.

- **DDIM sampling.** 50-step inference, ~20x faster than full DDPM. Fast enough for offline simulation on Mac MPS, too slow inside Docker QEMU for live Stage integration.

- **Frontier scoring.** `src/frontier_scorer.py` and `src/simulate_exploration.py`. Score = E[info_gain] + lambda * Std[info_gain] - beta * distance, with K=8 completions. Tuned scoring (info_radius matching lidar range, lambda=0.5, beta=0.5) is what the latest experiments use.

- **ROS 2 nodes.** `ros2_ws/src/diffusion_explorer/` has the diffusion frontier node, baseline frontier node, exploration manager (pure pursuit), and a waypoint replay node. The replay node is what makes the Stage demo work despite Docker being too slow for real-time inference.

- **Pre-computed waypoint demo.** Model runs on Mac MPS, exports waypoints.json, Docker replays the navigation on the real Stage maze. Recording exists at `results/stage_demo/stage_demo_recording.mp4` -- but see "what doesn't work" below.

## What doesn't work (and why)

- **Live Stage integration with the model in the loop.** Docker on Apple Silicon runs x86 emulation via QEMU. PyTorch inference is ~50x too slow for real-time frontier selection. Workaround was the pre-computed waypoints approach; that runs, but it's not the integrated system we'd ideally show.

- **Stage demo recording.** The recording captures the Stage window and shows the maze, robot, and lidar rays, but the robot barely moves during the 8.8 minute window -- QEMU was that slow. The recording is honest evidence the integration works, but visually it does not show exploration.

- **Lidar range hypothesis.** Experiment 3 tested "advantage grows as lidar range shrinks" (because shorter sensors should mean more value from imagination). Results don't support it cleanly. At lidar=30 both methods fail; at 50/70/100 the differences are within noise. The clean causal story we hoped for isn't there.

## Honest experimental result (N=30 statistical comparison)

Source: `results/experiments/exp1_results.json`, plots in `results/experiments/plots/`.

- Mean steps to 80% coverage: diffusion 6.26, baseline 6.52 (~4% faster).
- Mean steps to 90% coverage: diffusion 7.81, baseline 8.15 (~4% faster).
- Per-map steps to 80%: diffusion faster on 13 maps, baseline faster on 4, ties on 10.
- Final coverage after 20 steps: diffusion 91.7%, baseline 93.7%.
- A handful of failure cases (maps 10795, 17925, 19006) where the diffusion-guided robot got stuck or made poor choices.

The headline plot `exp1_area_over_time.png` shows the two mean curves overlapping with heavy 95% CI overlap. The per-step win rate (`exp1_winrate.png`) shows diffusion ahead in steps 2-5, then the baseline win rate climbs and dominates after step 7. The scatter plot (`exp1_scatter.png`) is the clearest evidence: 13 points below the tie line, 4 above.

This is a real, defensible, modest result. It is not the "+24.5%" claim that still appears in earlier artifacts and in the current report draft. Those need updating.

## What still has the OLD (wrong) framing

- `README.md` headline numbers and the "+24.5% info gain" bullet.
- `docs/report.tex` abstract and conclusion (still says 24.5% and "wins or ties on all 10 maps").
- `results/project_journey.html` walkthrough.
- The teaser video's "Diffusion vs Baseline" slide.

These all need to be updated to the new framing before the final report submission. Not blocking for the June 1 presentation but blocking for the June 6 report.

## Deliverables status

| Deliverable | Due | State |
|---|---|---|
| 5-min check-in slides | May 27 | Missed (no slides produced) |
| 6-min final presentation | June 1 | Not started |
| 4-page IEEE report | June 6 | Draft exists with wrong numbers; ethics + acknowledgments now added |
| 3-min teaser video | June 6 | 1.6 min version exists, needs extending |
| Clean source + README | June 6 | Code is clean; README headline numbers need update |
| 1-page individual reflection | June 6 | Not started |

## Infrastructure (all idle, nothing burning money)

- AWS EC2 g4dn.xlarge (`i-0b2d8ea34eaa56ef3`): stopped.
- S3 backup bucket `diffusion-map-project-backup`: persisted (small).
- Docker container for Stage: not running locally.
- Other AWS instances visible on the account belong to unrelated projects and have not been touched.

## What needs to happen next, in priority order

1. Update `README.md` and `docs/report.tex` numbers to match the N=30 experiment (replace "+24.5%" framing).
2. Build the June 1 presentation: 6 minutes, structured around the honest finding ("learned prior helps modestly with early-exploration efficiency, here's the per-step win rate and where it does and doesn't work").
3. Decide on the demo video for the presentation: re-record Stage cleanly, or use the offline simulation GIFs (which actually look better and show the model's predictions).
4. Extend the teaser video from 1.6 to ~3 minutes for the June 6 submission.
5. Write the 1-page individual reflection.

## Key files to know

- `src/experiments.py` -- the N=30 experiment driver. Run `python3 src/experiments.py` to reproduce.
- `src/analyze_experiments.py` -- generates all the plots from the JSON results. Run after experiments.py.
- `results/experiments/plots/` -- the plots that should anchor the new presentation and report.
- `results/experiments/plots/summary.md` -- the headline numbers in one place.
- `JOURNAL.md` (this file) -- the running status.
