# Presentation media bundle

All assets for tomorrow's FP3 talk in one place. Drag into Google Slides as needed.

## What's in each folder

### `slides/`
| File | What it is | Use |
|---|---|---|
| `slides.pptx` | 13-slide deck, Google Slides compatible | Upload to Drive → Open with → Google Slides |
| `slides.html` | Same deck in HTML, prints to PDF from Chrome | Primary deck if presenting from browser |
| `PRESENTATION_SCRIPT.md` | Speaker notes per slide with timing budget | Read once before the talk |

### `figures/` (static images, one or more per slide)
| File | Slide | What it shows |
|---|---|---|
| `slide03_data_samples.png` | 3 — The data | 20 HouseExpo (GT / partial / hidden) pairs |
| `slide05_training_loss.png` | 5 — Training curve | Train + val loss, plus IoU over 29 epochs |
| `slide06_sample_progression.png` | 6 — Sample progression | Model output at epochs 5/10/15/20/25 |
| `slide07_K_diversity.png` | 7 — K=8 diversity | Eight DDIM samples per partial map, multiple maps |
| `slide08_pipeline.png` | 8 — Pipeline | Architecture diagram |
| `slide10_4baseline_curves.png` | 10 — 4-baseline | Coverage curves for all 4 scorers |
| `slide10_4baseline_bars.png` | 10 — 4-baseline (alt) | Bar chart: step-4 vs final |
| `slide10_rollout_map2638.png` | 10 — Per-map evidence | 4-method side-by-side, big win |
| `slide10_rollout_map17666.png` | 10 — Backup | Second dramatic win |
| `slide10_rollout_map17925_failure.png` | 10 — Honest failure | All four methods plateau |
| `slide11_behind_mind_map2638.png` | 11 — Behind the mind | World / robot's view / model's mind, 4 steps |
| `slide11_behind_mind_map17666.png` | 11 — Backup | Same composite on a branching apartment |
| `slide12_OOD_ablation.png` | 12 — OOD ablation | In-domain vs OOD delta curves |
| `slide12_realtime_2x2_matrix.png` | 12 — Realtime upper bound | Prior × sampling 2x2 |
| `slide12_latency_profile.png` | 12 — Latency on T4 | K × DDIM steps timing |
| `backup_complexity_buckets.png` | Q&A | +4.7pp on hardest decile (N=80) |
| `backup_coverage_IoU.png` | Q&A | IoU vs scans (generalisation finding) |
| `backup_low_vs_high_coverage.png` | Q&A | 1-scan vs 5-scan prediction quality |

### `gifs/` (animations, can be inserted into Google Slides as images)
| File | Slide / use | What it shows |
|---|---|---|
| `00_SIDE_BY_SIDE_DEMO_diffusion_vs_baseline.gif` | **11 — primary demo** | **Stacked 2-row video: diffusion on top, baseline on bottom, same map, same start. The answer to "show how it would be like".** |
| `01_diffusion_loop_THE_ONE_FOR_PROF.gif` | 11 — main demo | Partial map + diffusion prediction + GT + coverage, animated |
| `02_baseline_loop_for_comparison.gif` | 11 — comparison | Same loop with the heuristic scorer instead of diffusion |
| `03_four_methods_synchronized.gif` | 11 — comparison | 2x2 panel, all four scorers on map 2638 in lockstep |
| `04_stage_simulator_ROS2.gif` | 12 — ROS proof | Diffusion-driven robot in Stage maze (PA3 world) |
| `05_IoU_climbs_with_scans.gif` | 6 — sample quality | Prediction sharpens as more scans accumulate |
| `06_exploration_extra.gif` | backup | Alternate exploration animation |
| `07_DDIM_denoising_process.gif` | 6 — diffusion explanation | DDIM denoising from pure noise to a floor plan |

### `videos/` (MP4 versions for video editors and the FP5 teaser)
| File | What it is |
|---|---|
| `SIDE_BY_SIDE_DEMO_diffusion_vs_baseline.mp4` | **Primary demo video (17 s). Diffusion stacked over baseline on the same map. Use this in the talk to show "the loop running".** |
| `stage_demo_short.mp4` | The original Stage demo (88 s) |
| `exploration_diffusion.mp4` | Diffusion exploration loop (17 s) |
| `exploration_baseline.mp4` | Baseline exploration loop (17 s) |
| `teaser_video.mp4` | 1.6-min draft for FP5 |

## How to use in Google Slides

1. Upload `slides.pptx` to Google Drive
2. Right-click → Open with → Google Slides (auto-converts)
3. For any slide where you want to swap an image: Insert → Image → Upload, pick from `figures/` or `gifs/`
4. GIFs play automatically in Google Slides presentation mode

## Total bundle size
About 18 MB.
