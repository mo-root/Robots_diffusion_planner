# Speaker Notes — Final Presentation (June 1, 2026)

**Total target:** 5:35 (leaves 25 seconds of slack for a 6:00 budget).
**Deck:** `minimal_deck.pptx` (10 slides).
**How to use:** copy the "Speaker note" block under each slide into the corresponding Google Slides speaker-notes box.

---

## Pre-presentation checklist

Before you walk to the front, confirm:

- [ ] Laptop connected to the projector, deck open in **Present mode** in Google Slides (or Chrome if HTML)
- [ ] All four demo GIFs visibly autoplay in present mode:
  - [ ] Slide 4 — `exploration_diffusion.gif` (main loop)
  - [ ] Slide 5 — `17_side_by_side_demo.gif` (diffusion vs baseline)
  - [ ] Slide 6 — `12_4baseline_curves.png` (static chart)
  - [ ] Slide 7 — `07_in_vs_ood_delta.png` + `11_2x2_matrix.png` (static charts)
- [ ] Backup of the deck on a USB stick or downloaded PDF
- [ ] Browser tab open with `slides_v2.html` as fallback in case Google Slides freezes
- [ ] Phone timer set to 6:00 — start it the moment you say your first word
- [ ] Water bottle nearby
- [ ] Speaker notes printed or open on phone (read them once on the way to class)

## Comparisons covered (what the prof asked for)

The prof asked for 4 frontier-scorer comparisons. **All four are in the deck:**

- ✅ **Nearest frontier** — argmin distance, shown in slide 6 stat card (+17.5pp gap)
- ✅ **Info-gain only** — no distance penalty, shown in slide 6 curves
- ✅ **Info-gain + distance** — conventional heuristic, shown in slide 6 + slide 5 demo
- ✅ **Diffusion (ours)** — K-sample UCB, shown in slide 6 + slide 4 demo + slide 5 demo

Additional asks covered:
- ✅ **Show the loop running** — slides 4 + 5 (two demo GIFs)
- ✅ **Inference timing** — slide 7 right panel + verbal in speaker note
- ✅ **Honest limitations** — slide 8 (Hardships)

## Timing schedule

| # | Slide | Allowance | Cumulative | Cue to advance |
|---|---|---|---|---|
| 1 | Title | 0:10 | 0:10 | After saying name + topic |
| 2 | Project Goal + Hypothesis | 1:00 | 1:10 | After finishing the three conditions |
| 3 | Approach — 4-card pipeline | 0:40 | 1:50 | After walking through all four cards |
| 4 | DEMO — the loop running | 0:35 | 2:25 | After one GIF loop completes |
| 5 | DEMO — side-by-side | 0:35 | 3:00 | After one GIF loop |
| 6 | Results — 4-baseline | 0:50 | 3:50 | After reading the three stat cards |
| 7 | When it fails + realtime | 0:45 | 4:35 | After both panels |
| 8 | Hardships | 0:30 | 5:05 | After the five bullets |
| 9 | What's Next | 0:25 | 5:30 | After the four bullets |
| 10 | Thanks | 0:05 | 5:35 | Pause for questions |

---

## SLIDE 1 — Title (0:10)

**Speaker note (copy into Google Slides):**

> "Hi everyone, I'm Moin. My project is **Diffusion-Guided Frontier Exploration** — I trained a diffusion model on real building floor plans and used it as a structural prior to help a robot decide where to explore next."

---

## SLIDE 2 — Project Goal + Hypothesis (1:00)

**Speaker note (copy into Google Slides):**

> "Think about walking into a friend's house you've never been to. You see a leash by the door, a chewed toy in the corner — and without thinking you predict there's probably dog food in the kitchen and a yard behind the back door. You don't just see what's there. You **imagine** what's behind the rooms you haven't entered, based on the structural cues you've picked up. That's the core idea behind **world models**: a learned prior that lets an agent predict the unseen.
>
> A robot doing frontier exploration with 2D lidar has none of that. The classical scorer picks based on unknown cells minus distance — it's **memoryless**. It throws away the fact that buildings have structure: walls run straight, hallways connect rooms, doors cluster.
>
> So I built the simplest possible version of a building world model: a diffusion model trained on 35,000 floor plans that the robot uses to imagine plausible completions of the unseen part, and score frontiers under those imagined buildings.
>
> The hypothesis is conditional: this prior gives a measurable early-budget head start, **but only when** **one**, the training distribution matches deployment, **two**, the step budget is tight, and **three**, inference is fast enough to drive. The rest of the talk tests each one."

---

## SLIDE 3 — Approach — End-to-End Pipeline (0:40)

**Speaker note (copy into Google Slides):**

> "Four-stage pipeline.
>
> **One — Data.** HouseExpo: 35,000 residential floor plans. For each one I generate a partial view from a simulated lidar scan and pair it with the hidden remainder. After flips and rotations, 2.66 million training pairs.
>
> **Two — Model.** Conditional U-Net, 4.16 million parameters. Standard DDPM, MSE on the noise. 29 epochs on a T4 GPU.
>
> **Three — Scoring.** At each frontier decision I draw K=8 DDIM completions. The frontier score is the expected information gain across those K samples, plus a variance bonus that acts as a UCB exploration signal, minus a distance penalty.
>
> **Four — Integration.** A ROS 2 node publishes to `/best_frontier`. The exploration manager from PA3 drives the robot in Stage."

---

## SLIDE 4 — DEMO: the loop running (0:35)

**Speaker note (copy into Google Slides):**

> "Let me show you the loop. **[Wait two seconds for the GIF to autoplay one cycle.]**
>
> Left panel — the partial map. Yellow dots are frontier candidates, orange is the top-scored, the green star is what gets picked. Middle — the diffusion model's prediction of the full building, with IoU updating. Right — ground truth, which the robot does not see. Far right — coverage climbing.
>
> What you're watching is the U-Net imagining a plausible building at every step, and the scorer using that imagination to pick the next frontier."

---

## SLIDE 5 — DEMO: diffusion vs baseline, same map (0:35)

**Speaker note (copy into Google Slides):**

> "Same map, but now two robots at once. **Top row** is diffusion-guided — my method. **Bottom row** is the classical heuristic baseline. Same start position, same lidar, same step budget.
>
> Watch the coverage tickers in the corners. Both methods get there eventually, but **diffusion is consistently a step or two ahead** in the early budget — it commits to the unseen wing faster because the prior tells it the building extends in that direction."

---

## SLIDE 6 — Results — 4-baseline ablation (0:50)

**Speaker note (copy into Google Slides):**

> "This is the headline experiment. **Four** frontier scorers, **30 held-out HouseExpo maps**, identical seeds.
>
> **Nearest frontier** — just argmin distance — gets 42.6% coverage at step four. **Info-gain alone** — argmax unknown cells, no distance penalty — jumps to 60.1%. **Info-gain plus distance**, which is the conventional heuristic, is **59.2%** — nearly identical, so the distance term contributes essentially nothing on residential maps. **Diffusion**, our method, gets 62.7%.
>
> Reading the deltas on the right: info-gain vs nearest is **plus 17.5 points** — that's where most of the value lives. Adding the distance term is roughly zero. The diffusion prior adds another **3.5 points** on top, concentrated in the early-budget regime.
>
> The decomposition is clean: info-gain matters, distance is decorative, the prior adds a small but real early-budget head start."

---

## SLIDE 7 — When it fails + what realtime would buy (0:45)

**Speaker note (copy into Google Slides):**

> "Two crucial controls.
>
> **Left — out-of-domain ablation.** I trained a second U-Net from scratch on synthetic warehouse layouts. Same architecture, same loss, completely different prior. Then I evaluated it on the **same** 30 residential maps. The advantage drops from +3.5pp to **negative 0.1pp** — statistically zero. That's the kill: it proves the prior is doing real structural work, not just averaging. A mismatched prior is no help.
>
> **Right — realtime upper bound.** Inference timing: K=8 with 30 DDIM steps is two seconds on a T4 — too slow. But K=4 with 10 DDIM steps is **500 milliseconds**, exactly the realtime budget. If we re-sampled every three driving cells — the speed a distilled model could achieve — the in-domain advantage extrapolates to **+10 points**.
>
> So: the prior matters, fast inference matters more. Distillation is the path."

---

## SLIDE 8 — Hardships + Honest limits (0:30)

**Speaker note (copy into Google Slides):**

> "Briefly, what didn't work.
>
> **Docker QEMU on Apple Silicon** was 50× too slow to run live diffusion inside Stage, so the Stage integration uses pre-computed waypoints. **Asymptotic coverage** — the baseline catches up by step 20; we don't beat it at the asymptote, only in the early budget. **Two of 30 maps** are hard failures where the model gets stuck — K-sample disagreement could detect this and fall back to heuristic. **Live hardware** is blocked by the same latency wall.
>
> Mismatched priors give zero help — the OOD experiment quantifies that."

---

## SLIDE 9 — What's Next (0:25)

**Speaker note (copy into Google Slides):**

> "Four next steps.
>
> **Distillation** to a Jetson-sized model for sub-second inference — that's the path to real-robot deployment.
>
> **Richer training domains** — Gibson, Matterport, real BIM data — for cross-environment generalisation.
>
> **K-sample disagreement** as an 'I don't know' flag — the variance can be a fallback trigger.
>
> **Live ROSbot deployment** in a matched-distribution building — search-and-rescue or hospital patrol would be the cleanest first targets."

---

## SLIDE 10 — Thanks (0:05)

**Speaker note (copy into Google Slides):**

> "Thanks for your attention. Happy to take questions."

---

## Q&A prep — likely questions + 30-second answers

### "Why diffusion instead of a single deterministic regression?"
> "Two reasons. One, a regression network would average all plausible completions into a blurry mush. Diffusion gives me crisp distinct samples. Two, I need the variance across samples as an exploration signal — that's only meaningful with a distribution. A regressor would give me a point estimate, no uncertainty."

### "Did you compare against any non-trivial baselines beyond nearest?"
> "Yes — four scorers total. Nearest, info-gain only, info-gain + distance, and ours. The 4-baseline ablation on slide 6 shows the decomposition: info-gain does most of the work, distance is decorative, the prior adds 3.5pp on top."

### "How does this generalize beyond HouseExpo?"
> "That's exactly what the OOD ablation tests. When I train on synthetic warehouse layouts and deploy on residential maps, the advantage drops to −0.1pp. So generalization needs distribution match. The next step is training on richer mixed domains — Gibson, Matterport — to broaden the prior."

### "Why is the Stage demo using pre-computed waypoints instead of running live?"
> "Docker QEMU on Apple Silicon adds a 50× latency penalty. With diffusion inference already at 2 seconds per decision on T4, running it inside emulated Stage would be 100 seconds per frontier — not usable. On real T4 hardware with K=4 and DDIM 10, latency drops to 500 milliseconds, which is what makes the realtime experiment a meaningful upper bound."

### "How is this different from Diffusion Forcing or world-model approaches?"
> "Good question. Diffusion Forcing is the more ambitious version — sequential prediction over multiple exploration timesteps, where the model predicts how the map evolves as the robot moves. What I built is the simpler single-step version: predict the full building from one partial map, repeat. Diffusion Forcing is in the 'what's next' slide as the natural extension."

### "What's your inference time on a real robot?"
> "Profiled on T4: K=8 / DDIM 30 is 2 seconds; K=4 / DDIM 10 is 500 ms. The 500 ms number is exactly the realtime budget — meaning a distilled model on a Jetson would fit the budget for a real ROSbot driving at typical exploration speeds."

### "What metric did you use for the +3.5pp number?"
> "Coverage at frontier-decision step 4, averaged over 30 held-out HouseExpo maps with identical seeds. I picked step 4 because that's where the advantage peaks in the curves — it's the early-budget regime. Asymptotic coverage at step 20 actually slightly favors the baseline."

### "What about ethics?"
> "Brief version — there's a dual-use risk. Same prior that helps search-and-rescue infer the layout of a collapsed building could let a surveillance robot infer a private floor plan from a brief glimpse. Mitigations include opt-in deployment, on-device redaction of bedroom/bathroom predictions, and broader training data audits. Full discussion is in the report."

---

## If a GIF doesn't play during the demo

Stay calm. Say:

> "The animation isn't loading — let me describe it. **[Use the still slide as context.]** What you'd see is the partial map on the left, the diffusion prediction in the middle, ground truth on the right, and coverage climbing on the far right. Each step the model samples K plausible buildings and the robot picks the frontier with the highest expected information gain plus variance bonus."

Then move on. Don't waste time fiddling.

---

## One-paragraph fallback (if everything fails)

If the laptop crashes mid-talk, deliver this verbally:

> "My project is a frontier exploration system that uses a conditional diffusion model trained on 35,000 HouseExpo floor plans as a structural prior. At each frontier decision the robot samples K=8 plausible completions of the unseen part and scores each candidate frontier by expected information gain plus variance bonus minus distance. On 30 held-out maps this gives a 3.5-point coverage advantage at step 4, growing to 4.7 points on the hardest decile. The OOD ablation — warehouse-trained model on residential maps — drops the advantage to −0.1 points, which proves the prior is doing real structural work. If inference were realtime (which the latency profile shows is achievable on T4 at K=4 / DDIM 10) the advantage extrapolates to +10 points. Stage integration runs via pre-computed waypoints because Docker QEMU on Apple Silicon is too slow for live diffusion. Future work is distillation to Jetson, richer training domains, and live ROSbot deployment."

---

**Good luck. You've got this.**
