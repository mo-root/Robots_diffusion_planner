# Final Presentation Outline — June 1, 2026
## Diffusion-Guided Frontier Exploration — Moin Mattar — COSC 81/281

**Format:** 6 minutes, 13 slides, ~28 seconds per slide on average.
**Visual style:** mirrors your earlier presentation rhythm — section dividers, bullets, images, memes.
**Drop-in instructions:** copy each slide block below into a fresh Google Slides slide.
**All assets are in `presentation_media/` (already zipped).**

---

## ACT 1 — Novelty & Motivation (4 slides, 1:30 total)

---

### Slide 1 — Section divider

**Layout:** like your slide 1 in the earlier deck

> # 1  |  Novelty & Motivation
> _Robots that imagine the rooms they have not seen yet._

**Speaker note (10 s):** "I am Moin Mattar. This is a frontier exploration project that uses a generative model of building layouts as a structural prior."

---

### Slide 2 — Project Goal

**Title:** Project Goal

**Bullets (left side):**
- **Problem:** a robot is dropped at the door of a building it has never seen. It has 2D lidar. After each ping it has a slightly bigger map.
- **Inputs:** partial occupancy grid (256×256), robot pose.
- **Outputs:** the next frontier to drive to.
- **Assumption:** deployment environment shares structural distribution with training data.
- **Why it matters:** budget-constrained exploration is the regime where battery, time, or danger windows actually bite.

**Image (right side):** a partial map next to the same map filled in.
- File: `presentation_media/figures/slide03_data_samples.png` (just crop one tile)

**Speaker note (20 s):** "The classical baseline scores frontiers by unknown cells minus a distance penalty. It is memoryless — it throws away everything we know about buildings. Walls run straight, hallways connect rooms, doors cluster. Our hypothesis: a generative model of real floor plans can use that structure."

---

### Slide 3 — Novelty?

**Title:** Novelty?

**Bullets:**
- We treat **diffusion samples** as imagined buildings, not just predictions.
- We use **K-sample disagreement** as an upper-confidence-bound exploration signal — the variance is the feature.
- We isolate **where the wins come from** with a 4-baseline ablation that decomposes the conventional pipeline.
- We add an **out-of-domain control** that confirms the prior is doing real work, not averaging noise.

**Image:** any "thinking" image / meme from your stash, or use the Descartes joke from your earlier deck.

**Speaker note (20 s):** "The novelty is not the U-Net — it's the framing. Diffusion gives us a sampler over plausible buildings. Variance across K samples is exploration uncertainty. We decompose the standard frontier scorer to show what each term is actually contributing."

---

### Slide 4 — Hypothesis (boxed claim like your earlier slide 5)

**Title:** Hypothesis

**Boxed text (left, green left-border like your earlier slide):**
> A learned generative prior of building layouts gives a robot a measurable early-budget head start in frontier exploration — but **only** when (1) training distribution matches deployment, (2) the step budget is tight, and (3) inference is fast enough to drive.

**Image (right):** robot-thinking meme or the "Bro really thinking" image from your earlier deck.

**Speaker note (20 s):** "Three conditions. The deeper experiments later in the talk will test each of these independently."

---

## ACT 2 — Our System (4 slides, 1:50 total)

---

### Slide 5 — Section divider

> # 2  |  Our System
> _HouseExpo → Conditional U-Net (DDPM) → K-sample scoring → frontier pick_

**Speaker note (10 s):** "Here is what I actually built."

---

### Slide 6 — End-to-End Pipeline (4-card grid, like your earlier slide 9)

**Title:** End-to-End Pipeline I Implemented

**Four colored cards:**
1. **Data Collection** (green) — HouseExpo, 35k plans, rasterised, 2.66M training pairs after augmentation.
2. **U-Net + DDPM** (orange) — conditional U-Net, 4.16M params, MSE on noise, T4 GPU, 29 epochs.
3. **K-Sample Scoring** (yellow) — K=8 DDIM completions, score = E[gain] + λ·Std[gain] − β·dist.
4. **Stage Integration** (blue) — ROS 2 node publishing `/best_frontier`, drives `exploration_manager` in Stage maze.

**Speaker note (20 s):** "Four stages. Data, model, scoring, robot. The middle two — the diffusion sampling and the UCB-style score — are the new logic beyond the PA assignments."

---

### Slide 7 — The Data (table + sample grid, like your earlier slide 10)

**Title:** The Data

**Table:**

| Source | Plans | Pairs after augmentation | Purpose |
|---|---|---|---|
| HouseExpo (residential) | 35,126 | 2.66M | training |
| Held-out HouseExpo | 30 | per-rollout | evaluation |
| Synthetic warehouse | 8 epochs | mini-set | OOD control |

**Below table:** *Design choice — augmentation (4 rotations + flips) was the dominant lever for sample diversity.*

**Image:** `presentation_media/figures/slide03_data_samples.png`

**Speaker note (15 s):** "35,000 floor plans, augmented to 2.66 million pairs. Each pair is a partial view from a random lidar scan and the hidden remainder."

---

### Slide 8 — Baseline Objective (math, like your earlier slide 15)

**Title:** Baseline Objective (DDPM)

**Math (centered):**

Forward process:
$$x_\tau = \sqrt{\bar{\alpha}_\tau} x_0 + \sqrt{1 - \bar{\alpha}_\tau} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

One-step objective:
$$\mathcal{L}_{\text{base}} = \mathbb{E}\left[ \|\hat{\epsilon}_\theta(x_\tau, \tau, \text{partial}) - \epsilon\|_2^2 \right]$$

**Bullets:**
- Works for single-step denoising.
- Conditioned on the partial occupancy grid and a known-mask channel.
- DDIM sampling at inference (30 steps, K=8 samples per batched call).

**Speaker note (15 s):** "Standard DDPM, MSE on the noise. Two conditioning channels: the partial map and the known-mask. At inference I use DDIM to drop the sample cost from 1000 steps to 30."

---

### Slide 9 — Frontier Scoring (the actual new logic)

**Title:** Scoring frontiers under K imagined buildings

**Math (centered):**

For each frontier cluster $f$:
$$\text{score}(f) = \mathbb{E}_{k=1\ldots K}[G_k(f)] + \lambda \cdot \text{Std}_{k=1\ldots K}[G_k(f)] - \beta \cdot d(\text{robot}, f)$$

where $G_k(f)$ = unknown cells inside frontier sensor footprint under completion $k$.

**Bullets:**
- **Mean gain** = how much the robot expects to discover at this frontier.
- **Variance bonus** = UCB-style exploration term; the model's disagreement.
- **Distance term** = travel cost.
- Pick the argmax. Move. Sense. Re-sample. Repeat.

**Image (optional):** `presentation_media/figures/slide07_K_diversity.png` (smaller, inset)

**Speaker note (15 s):** "This is the actual new logic. Standard frontier scorers average over what they have seen. We average over K imagined buildings, plus we add a variance term so the robot is pulled toward frontiers where the model itself is unsure."

---

## ACT 3 — Results (4 slides, 2:00 total)

---

### Slide 10 — Now you tell me (4-baseline ablation strip)

**Title:** Now you tell me  (or: "Where do the wins come from?")

**Comparison labels (top of image):**  Nearest  |  Info-gain only  |  Info-gain + dist  |  **Diffusion (ours)**

**Image (large):** `presentation_media/figures/slide10_4baseline_curves.png`

**Side stats (3 cards):**
- **+3.5pp** — diffusion vs heuristic, step 4
- **+17.5pp** — info-gain vs nearest, step 4
- **~0pp** — distance term contribution

**Speaker note (25 s):** "I ran four scorers on the same 30 HouseExpo maps with identical seeds. The decomposition is clean: info-gain is doing most of the work; the distance term is decorative on HouseExpo; the learned prior adds another 3.5 points on top, concentrated in the early-budget regime."

---

### Slide 11 — Behind the mind (the loop made visible)

**Title:** Behind the mind  (world → observation → imagination → pick)

**Image (large, full slide):** `presentation_media/figures/slide11_behind_mind_map2638.png`

**Caption below:** *Top: the world the robot doesn't see, with its trail. Middle: the partial map and frontier candidates. Bottom: the mean of K=4 diffusion samples (the imagined building), with the chosen frontier circled.*

**Speaker note (20 s):** "Same map, same lidar, four steps. On map 2638 the prior carries the robot to 95% coverage by step 4. The information-gain methods plateau at 74% because they can't see past a doorway the prior expects to exist."

---

### Slide 12 — DEMO (side-by-side animated, like your "Now you tell me" + "DEMO?" combined)

**Title:** DEMO — Diffusion vs Baseline, same map

**Image (insert as animated GIF in Google Slides):**
`presentation_media/gifs/00_SIDE_BY_SIDE_DEMO_diffusion_vs_baseline.gif`

**Caption below:** *Top row: diffusion-guided. Bottom row: heuristic baseline. Same map, same start, same lidar. Both reach high coverage; diffusion is consistently ahead in the early steps.*

**Speaker note (20 s):** "This is the loop running. Top row is diffusion, bottom is baseline. Watch the coverage tickers in the corners. Both methods get there, but diffusion commits to the unseen wing one or two steps earlier."

---

### Slide 13 — Hardships  +  Honest limits  (a Hardships slide like your earlier slide 22)

**Title:** Hardships

**Bullets (two columns):**

**Compute / infra:**
- Docker QEMU on Apple Silicon: too slow for live diffusion in Stage; switched to pre-computed waypoints.
- AWS EC2 g4dn.xlarge (T4 GPU) for training and inference experiments.
- Batched K=8 sampling rewrite: 3× speedup, no quality loss.

**Honest limits:**
- **Asymptotic coverage** — baseline catches up by step 20.
- **Out-of-domain priors** — warehouse-trained model on residential maps gives **−0.1pp** (the kill).
- **2 of 30 hard failures** — K-sample disagreement could flag these.
- **Live hardware** — Stage works via pre-computed waypoints, not live inference.

**Image (optional):** `presentation_media/figures/slide12_OOD_ablation.png` (small, inset on the right)

**Speaker note (25 s):** "The honest framing. The Stage integration runs via pre-computed waypoints because Docker QEMU emulation is too slow for live diffusion. The out-of-domain experiment is the cleanest negative control: when you train on warehouses and deploy on houses, the prior is statistically zero help. That confirms the prior is doing real structural work, not just averaging."

---

## CLOSE — Sampling + What's Next + Thanks (2 slides, 0:40 total)

---

### Slide 14 — Sampling timing  +  What's Next (combined, like your earlier slide 23 + 26)

**Title:** Sampling timing  +  What's next

**Left side — timing (per your earlier deck):**
- DDIM 30 steps, K=8: **~2 s** on T4
- DDIM 10 steps, K=4: **~500 ms** on T4 — exactly the realtime budget
- Realtime re-sampling lifts in-domain advantage from +3.5pp to **+10.0pp** (extrapolated)

**Right side — what's next:**
- **Distillation** to a smaller U-Net on Jetson for sub-second inference on a real robot.
- **Train on richer domains** — Gibson, Matterport, real BIM data.
- **K-sample disagreement** as an "I don't know" flag — fall back to heuristic on hard maps.
- **Diffusion Forcing for maps** — sequential map prediction over multiple exploration timesteps (future work direction).

**Speaker note (20 s):** "Timing is the gating concern for real robots. Latency profiling shows 500 milliseconds is achievable on T4. Distillation puts the realtime+10-point gain in reach for a Jetson on a ROSbot."

---

### Slide 15 — Thanks for your attention

**Title:** Thanks for your attention. Questions?

**Bottom signature:** Moin Mattar  ·  COSC 81/281  ·  AI helped me in formatting and writing (HTML/CSS/LaTeX), as well as explained concepts.

**Image:** any closer meme — your earlier Leonardo DiCaprio "now you can clap" works perfectly.

**Speaker note (5 s):** "Thank you. I'm happy to take questions."

---

## TIMING BUDGET (target 6:00)

| # | Slide | Allowance | Cumulative |
|---|---|---|---|
| 1 | Section divider — Novelty | 10 s | 0:10 |
| 2 | Project Goal | 20 s | 0:30 |
| 3 | Novelty? | 20 s | 0:50 |
| 4 | Hypothesis | 20 s | 1:10 |
| 5 | Section divider — Our System | 10 s | 1:20 |
| 6 | End-to-End Pipeline | 20 s | 1:40 |
| 7 | The Data | 15 s | 1:55 |
| 8 | Baseline Objective (math) | 15 s | 2:10 |
| 9 | Frontier Scoring (math) | 15 s | 2:25 |
| 10 | 4-baseline ablation | 25 s | 2:50 |
| 11 | Behind the mind | 20 s | 3:10 |
| 12 | DEMO side-by-side | 35 s _(let the GIF play 1.5 cycles)_ | 3:45 |
| 13 | Hardships + honest limits | 25 s | 4:10 |
| 14 | Timing + What's next | 20 s | 4:30 |
| 15 | Thanks + questions | 5 s | 4:35 |

Total: **4:35** — leaves you **~85 seconds of slack** for Q&A or for slowing down on the math slides if you want.

---

## ASSETS CHECKLIST (all already in `presentation_media/`)

**Images to drag into Google Slides:**
- `figures/slide03_data_samples.png` → Slide 7 (Data)
- `figures/slide07_K_diversity.png` → Slide 9 inset (optional)
- `figures/slide10_4baseline_curves.png` → Slide 10 main image
- `figures/slide11_behind_mind_map2638.png` → Slide 11 main image
- `figures/slide12_OOD_ablation.png` → Slide 13 inset (optional)

**Animated GIFs (Google Slides plays them in present mode):**
- `gifs/00_SIDE_BY_SIDE_DEMO_diffusion_vs_baseline.gif` → Slide 12 main image (DEMO)
- `gifs/01_diffusion_loop_THE_ONE_FOR_PROF.gif` → backup if side-by-side doesn't play cleanly

**Memes / personality (your call):**
- Slide 3 (Novelty?): pick anything from your previous deck or use a fresh one
- Slide 4 (Hypothesis): "Bro really thinking" image works
- Slide 15 (Thanks): Leonardo "now you can clap"

---

## HOW TO ASSEMBLE IN GOOGLE SLIDES

1. Open Google Slides → File → New → blank presentation.
2. Slide → Apply layout → "Blank" for every slide.
3. For each slide above:
   - Insert → Text box → paste the title in bold (40-48 pt).
   - Insert → Text box → paste the bullets (18-22 pt).
   - Insert → Image → Upload from computer → pick the asset listed.
4. For the DEMO slide (12): Insert → Image → upload the GIF — Google Slides plays GIFs in present mode automatically.
5. Set slide size to 16:9 (File → Page setup) if not already.
6. Test in Present mode. The GIF on slide 12 should auto-play.

---

## WHAT THIS DELIVERS AGAINST THE CLASS RUBRIC

| Rubric criterion | Where it lands in the deck |
|---|---|
| Problem Statement (inputs / outputs / assumptions) | Slide 2 |
| Task Planning (high-level reasoning, not reactive) | Slide 9 (K-sample scoring) |
| Implementation (substantial new logic) | Slides 6, 8, 9 (pipeline, DDPM, scorer) |
| Synthesis (sub-systems coupled) | Slide 6 (4-card pipeline shows coupling) |
| Demonstration (video/sim demo) | Slide 12 (side-by-side GIF) |
| Honest limitations | Slide 13 (hardships + OOD kill + asymptotic) |
| Form (slide quality + time) | Whole deck, 4:35 with slack |

---

## ONE-PARAGRAPH PITCH (memorise this — useful for Q&A)

> "A robot in an unknown building has to keep picking which frontier to drive to next. The classical heuristic scores frontiers by unknown cells minus distance — it's memoryless. I trained a conditional diffusion U-Net on 35,000 HouseExpo floor plans, and at each frontier decision I sample K=8 plausible completions of the unseen part. The frontier score becomes the mean expected information gain plus the variance across K samples as an exploration bonus, minus distance. On 30 held-out maps, this gives a +3.5-point early-budget advantage at step 4. The advantage grows to +4.7 on the hardest decile and vanishes to −0.1 with a mismatched prior — a clean negative control that confirms the prior is doing real structural work. If inference were sub-second, the realtime advantage extrapolates to +10 points. The Stage integration runs via pre-computed waypoints because Docker QEMU on Apple Silicon was too slow for live diffusion."

That's the whole talk in one paragraph. If anything goes wrong with the slides, you can just say this.
