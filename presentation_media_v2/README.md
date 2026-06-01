# Final Presentation Media Bundle (v2)

**18 slides · 6 minutes · matches Phase 3 class requirements (problem / approach / demo).**

## How to use this bundle

### Option A — open the HTML deck directly
1. Open `slides_html_pptx/slides.html` in Chrome
2. Arrow keys to advance, **P** to print to PDF for Gradescope upload
3. All GIFs autoplay in browser

### Option B — drag assets into Google Slides
Each folder below is named after the slide it belongs to. Drag the contents into the matching slide.

## Folder-to-slide mapping

| Folder | Slide | What it shows |
|---|---|---|
| `slide_03_baseline_loop/` | 3 — Project goal | `exploration_baseline.gif` — the classical loop (no prediction in middle panel) |
| `slide_06_pipeline/` | 6 — Pipeline (text-only, no media) | — |
| `slide_07_data/` | 7 — The data | `01_sample_grid_20.png` — 20 HouseExpo training pairs |
| `slide_08_training/` | 8 — Training curves | `fig_loss.png` — train/val loss + IoU over 29 epochs |
| `slide_09_denoising/` | 9 — How the model thinks | `02_denoising_process.gif` — DDIM denoising noise → building |
| `slide_10_K_diversity/` | 10 — K=8 diversity | `03_sample_diversity.png` — 8 samples per partial input |
| `slide_12_demo_main_loop/` | 12 — DEMO: end-to-end loop | `THE_ONE_FOR_PROF.gif` — the headline animation + `exploration_diffusion.mp4` |
| `slide_13_demo_side_by_side/` | 13 — DEMO: diffusion vs baseline | `17_side_by_side_demo.gif` + `.mp4` — stacked comparison |
| `slide_14_demo_four_methods/` | 14 — DEMO: 4 methods synchronized | `16_4method_sync.gif` — 2×2 panel, all four scorers |
| `slide_15_behind_mind_and_stage/` | 15 — Behind the mind + Stage | `15_behind_mind_map2638.png` + `maze_exploration.gif` |
| `slide_16_results_4baseline/` | 16 — 4-baseline results | `12_4baseline_curves.png` (main) + `per_map_evidence.png` (backup) |
| `slide_17_OOD_kill/` | 17 — Hardships + OOD + realtime | `07_in_vs_ood_delta.png` + `realtime_2x2_matrix.png` |
| `slides_html_pptx/` | — | `slides.html` (primary), `intro_slides_2slide.pptx` (intro only), `slides_old.pptx` (older 13-slide version) |

## Slide-by-slide timing budget (target 6:00)

| # | Slide | Time | Cumulative |
|---|---|---|---|
| 1 | Title | 10 s | 0:10 |
| 2 | Section divider — Problem | 5 s | 0:15 |
| 3 | Project Goal | 25 s | 0:40 |
| 4 | Novelty + Hypothesis | 30 s | 1:10 |
| 5 | Section divider — Approach | 5 s | 1:15 |
| 6 | End-to-End Pipeline | 25 s | 1:40 |
| 7 | The Data | 15 s | 1:55 |
| 8 | Training curve | 15 s | 2:10 |
| 9 | Denoising animation | 20 s | 2:30 |
| 10 | K=8 diversity | 20 s | 2:50 |
| 11 | Section divider — Demo | 5 s | 2:55 |
| 12 | DEMO main loop | 30 s _(let GIF play 1+ cycles)_ | 3:25 |
| 13 | DEMO side-by-side | 30 s | 3:55 |
| 14 | DEMO 4 methods | 25 s | 4:20 |
| 15 | Behind the mind + Stage | 25 s | 4:45 |
| 16 | 4-baseline results | 30 s | 5:15 |
| 17 | Hardships + OOD + realtime | 30 s | 5:45 |
| 18 | Close + thank you | 15 s | 6:00 |

**Total: exactly 6:00** with no overrun.

## What the class rubric wants vs what this deck delivers

| Class requirement | Slide(s) |
|---|---|
| Clearly state the problem | 3 |
| Describe the approach | 4–10 |
| Live demo or video | 12, 13, 14, 15 (four different demos) |
| Honest limitations | 17 |
| Effective use of time | ~6:00 budget with 4 dividers for pacing |

## Bundle stats
- 18 slides
- 13 media files (~10 MB)
- All paths in `slides.html` are relative — they resolve to `/Users/moin/Robots_diffusion_planner/results/...`
- For the PPTX path, the file references are embedded directly inside the PPTX

## Quick prep checklist before tomorrow

- [ ] Open `slides.html` in Chrome → walk all 18 slides → verify every GIF loads and autoplays
- [ ] Print to PDF (Chrome ⌘P → landscape → save) for Gradescope
- [ ] Upload `slides_old.pptx` to Google Drive as a Google Slides backup
- [ ] Practice the talk once — aim for 5:45 to leave 15 s slack
