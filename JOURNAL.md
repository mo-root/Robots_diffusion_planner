# Project Journal

A running log of where this project stands. Read this first if you're returning to it after a break.

## Current state (as of 2026-05-30)

The model is trained, the offline pipeline works, the ROS 2 integration runs in Stage, and we have a rigorous N=30 experiment quantifying whether learned map priors actually help frontier exploration. The honest answer is "modestly, on most maps, mostly in the early steps." That story replaces an earlier, misleading "+24.5% info gain" claim that conflated single-frontier scoring with full exploration efficiency.

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
