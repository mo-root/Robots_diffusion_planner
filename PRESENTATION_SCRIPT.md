# Diffusion-Guided Frontier Exploration — Six-Minute Script

Status: working draft as of 2026-05-31. Numbers update as experiments land.
Final-N results expected from box 1 (Exp A complexity + Exp B N=100) and
box 2 (Exp D realtime) within ~90 minutes.

---

## Slide 1 — Title (15 sec)

"Diffusion-Guided Frontier Exploration. I am Moin Mattar. The question I am going
to answer in six minutes is whether a learned generative model of building layouts
can help a robot choose where to drive next when it is exploring an unknown
floor — and just as importantly, **when it cannot**."

---

## Slide 2 — The setup (45 sec)

"Drop a robot at the front door of a building it has never seen. It has a 2D lidar
that sweeps out and returns distances. After each lidar ping it has a slightly
bigger map. The question is always the same: pick the next frontier."

"The classical approach scores every frontier cell by the unknown cells in its
neighborhood, minus a distance penalty. It is purely memoryless. It throws away
the fact that real buildings have walls that run straight, hallways that connect
rooms, doors that cluster."

"Our hypothesis: a generative model trained on real floor plans can imagine
plausible completions of the unseen part, and use those imagined cells to score
frontiers under both expected value and uncertainty."

---

## Slide 3 — The pipeline (45 sec)

"Partial map goes in. A conditional U-Net diffusion model with 4 million parameters
samples K equals 8 plausible full-building completions in one batched DDIM pass.
For each candidate frontier, we count expected lidar gain across the K imagined
buildings, add the standard deviation as an upper-confidence-bound term, subtract
travel distance, and pick the argmax."

"Same scoring loop as the baseline. Only difference: the frontier neighborhood is
evaluated against K imagined completions instead of the raw partial map."

---

## Slide 4 — Behind the mind (40 sec)

"Before showing the results, here's the loop made concrete. Same map, same start, four columns at steps 1, 2, 4, 8."

"Top row: the world the robot doesn't see, with its trail traced over it. Middle row: the partial occupancy grid the robot has actually accumulated, with frontier candidates plotted in blue. Bottom row: the mean of four diffusion samples — the model's imagined building — with the chosen frontier circled."

"By step 4, the robot has imagined a near-complete building, and it's already committed to driving into the unseen wing. By step 8 the imagination matches reality and the loop closes."

[Show figure 15_behind_mind_map2638.png]

---

## Slide 5 — Result 0: where the wins come from (45 sec) — **4-baseline ablation**

"Before showing the headline result, here's what's actually doing the work. I ran four frontier scorers on the same 30 HouseExpo maps with identical seeds: nearest-frontier, info-gain only, info-gain plus distance (the conventional heuristic), and our diffusion scorer."

"At step 4, nearest gets 42.6% coverage. Info-gain alone gets 60.1%. Adding the distance penalty on top — 59.2% — basically zero contribution. So the standard exploration heuristic is essentially the info-gain term. The distance penalty is decorative."

"Diffusion gets 62.7% at step 4 — a +3.5pp lift over the conventional heuristic, and +17.5pp over nearest. That's the clean decomposition: info-gain matters, distance does nothing on residential floor plans, and the learned prior adds a measurable early-budget head start on top."

[Show figure 12_4baseline_curves.png]

---

## Slide 6 — Per-map evidence: map 2638 (25 sec)

"To make that concrete: same map, same start position, same lidar, same 20-step budget. On map 2638, by step 4 the diffusion scorer has already reached 95% coverage. Info-gain methods plateau at 74% — they cannot find their way past a doorway the prior expects. Nearest catches up by step 8."

[Show figure 14_rollout_map2638.png]

"This is one map of 30. The averaged numbers above include both wins like this and the failure cases I'll cover at the end."

---

## Slide 7 — Result 1: in-domain by complexity (40 sec)

"30 random floor plans from held-out HouseExpo. Same start position, same lidar,
same step limit, no cherry-picking."

"At step 4, our method has +3.5% absolute coverage advantage over the baseline,
18 of 30 maps win, 8 lose, 4 tie. The advantage peaks around steps 2 to 5 — the
regime that matters when a robot has a limited budget."

"To stress-test this, we re-ran on 80 maps and stratified by complexity
(initial frontier count, a clean proxy for branching factor). On the simplest
maps the advantage is +2.6%. On the most complex it doubles to **+4.7%** with
wins outnumbering losses 4.5 to 1. **The harder the map, the more the prior
matters.**"

[Show figure 04_complexity_buckets.png]

[Backup: 01_houseexpo_n30_delta.png and 02_houseexpo_n30_wins.png]

---

## Slide 8 — Result 2: out-of-domain ablation (45 sec) — **the kill**

"Here is the question reviewers always ask. Is the prior actually helping, or is
the U-Net just a fancy averaging operator?"

"We trained a SECOND model on synthetic stereotyped environments — grids of rooms
and warehouse aisles — for 8 epochs in 6 minutes on a T4. Then evaluated it on
the SAME 30 HouseExpo maps with the SAME scoring code."

"Result: the +3.5% advantage vanishes. Step-4 delta drops to −0.1%. Wins go from
18 down to 9, losses up to 12. The OOD prior is basically zero help."

[Show figure 07_in_vs_ood_delta.png]

"This is a clean negative control. The prior is not magic. It only contributes
useful information when the training distribution matches deployment."

---

## Slide 9 — Result 3: realtime upper bound (45 sec)

"One real-world objection: our model takes 7 seconds per decision on a T4. That
is too slow to drive a real robot, where each frontier decision needs to happen
in well under a second."

"So we ran a third experiment: assume free inference. Re-sample K=8 completions
every 3 driving steps instead of only at frontier decisions. This is what would
happen if we distilled the model down to sub-second latency on a Jetson."

"Result on N=20 per cell: the full 2x2 matrix of prior × sampling mode.
In-domain + realtime: **+10.0% over baseline**. OOD + realtime: **+6.0%**.
In-domain + discrete: +3.5%. OOD + discrete: −0.1%."

"This lets us decompose the two contributions. Prior match alone is worth about
+4%. Sampling frequency alone is worth about +7%. They are roughly additive,
and **sampling frequency matters more than domain match in our data**. Even
with the wrong prior, fast re-sampling rescues most of the advantage."

"Practical translation: distilling the model to sub-second inference would
unlock a bigger head-start than getting the training distribution perfect. That
is the strongest argument for hardware deployment we can make."

---

## Slide 10 — Where this is useful (30 sec)

"The structure-and-domain thesis says this approach is useful when **three**
conditions all hold:"

"1. The deployment environment is structured and known in advance — warehouses,
hospitals, office buildings, mines, hotels. A model trained on the right
distribution gives a real head-start."

"2. The robot has a tight budget — battery, time, danger window. The advantage
is concentrated in the first 5 frontier decisions, which is exactly the regime
budget-constrained robots live in."

"3. We can run inference fast enough to actually drive — sub-second. The
realtime experiment shows the upper bound is meaningfully higher than the
per-decision mode we tested."

"Search-and-rescue is the cleanest fit. So is patrol-style inspection in known
building classes. Generic exploration of arbitrary spaces is not."

---

## Slide 11 — Honest limits + next (20 sec)

"What we cannot claim:"
"- Asymptotic coverage. Baseline catches up. We do not beat it at step 20."
"- Hardware deployment. ROS 2 + Stage integration works via pre-computed waypoints
  because Docker QEMU on Apple Silicon is 50x too slow for live inference."
"- Generic exploration. Without a matching prior, the K-sample expectations are
  noise."
"- Two of 30 maps were hard failures where our model got stuck (a third was a
  both-methods failure, really a planner issue). 7% hard-failure rate is not
  zero."

"What is next, concretely:"
"- We measured latency on T4. With K=4 samples and 10 DDIM steps, inference is
  500 milliseconds. That is exactly the realtime budget. Distillation to a smaller
  U-Net on Jetson would put the +10% advantage in reach for real robots."
"- Train on richer environment classes (Gibson, Matterport, real BIM data)."
"- Use K-sample disagreement to detect 'I don't know' and fall back to baseline."
"- Run on real hardware in a matched-distribution building."

"That is the project. Thank you. Questions."

---

## Backup slides (for Q&A)

- The K-sample uncertainty term — how λ trades off variance.
- Per-map example: map 17666, baseline 9 frontier decisions to 80%, ours 5.
- Failure modes: 3 of 30 maps where our diffusion robot got stuck (uncertainty
  estimation could detect and fall back to baseline).
- Why diffusion and not GANs/VAEs: K diverse sharp samples per call, uncertainty
  comes free from K-sample disagreement.

## Timing budget

| Slide | Allowance | Cumulative |
|---|---|---|
| 1 Title | 15s | 0:15 |
| 2 Setup | 30s | 0:45 |
| 3 Pipeline | 30s | 1:15 |
| 4 Behind the mind | 40s | 1:55 |
| 5 4-baseline ablation | 45s | 2:40 |
| 6 Per-map rollout (map 2638) | 25s | 3:05 |
| 7 In-domain by complexity | 40s | 3:45 |
| 8 OOD ablation | 45s | 4:30 |
| 9 Realtime upper bound | 45s | 5:15 |
| 10 Where useful | 30s | 5:45 |
| 11 Limits + next | 15s | 6:00 |

## Open questions still being answered

1. **Complexity stratification** (Exp A on box 1): does the in-domain advantage
   grow with map complexity? If yes, add to slide 4. If no, leave it out.
2. **Realtime mode** (Exp D on box 2): how much does free-inference help?
   Number goes in slide 6.
3. **N=100 robustness** (Exp B on box 1): does the +3.5% finding hold with
   tighter CIs? If yes, update slide 4 numbers.

---

## Speaker notes

- The cross-domain ablation is the single most important thing to land cleanly.
  It separates this from a vague "we tried diffusion, it kind of helped" project.
- Do not overclaim. The honest framing — modest in-domain advantage, zero OOD
  advantage — is more defensible than any inflated number would be.
- If audience asks "why did you not test on more environments?" — answer:
  "Synthetic environments hit a planner limitation we did not have time to fix.
  The cross-domain ablation we did run is the cleaner test anyway."
