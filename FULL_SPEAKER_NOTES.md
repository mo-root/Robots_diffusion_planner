# Speaker notes for the project_journey page

Word-for-word, card by card. Matches what is actually on the page right now.

Open the page in Chrome. Scroll as you talk. Each block below corresponds to one card on the page.

**Format key:**
- `[POINT AT ...]` is an action, do not say out loud
- Quoted bold = what to say
- Numbers in the right column = cumulative time

---

## HERO  (0:00 → 0:10)

`[Page open at the top, "Diffusion-Guided Frontier Exploration" hero is on screen]`

> **"Hi, I am Moin. My project is Diffusion-Guided Frontier Exploration. I trained a diffusion model on real building floor plans and used it as a structural prior to help a robot decide where to explore next."**

`[Scroll down past the nav into Section 0]`

---

## SECTION 0 — INSPIRATION: WORLD MODELS  (0:10 → 0:55)

### Card: Related work table

`[The Related work table is on screen]`

> **"The biggest direction in AI research right now is world models. Generative models that learn how the physical world is structured so an agent can imagine and predict the unseen."**

> **"Think about walking into a friend's house you have never been to. You see a leash by the door and a chewed toy in the corner, and without thinking you already predict there is probably dog food in the kitchen. You imagine what is behind the rooms you have not entered."**

`[POINT AT the table]`

> **"This is happening at scale. NVIDIA Cosmos trained world foundation models on 20 million hours of robot video. DeepMind Genie 2 generates interactive 3D environments from a single image. Decart shows real time generative world models on commodity GPUs are now possible. Diffusion Forcing is the sequence-prediction extension I list as future work."**

> **"My project is the simplest possible instance of this paradigm for a class robotics problem."**

### Callout: Hypothesis

`[Scroll to the green hypothesis callout]`

> **"My hypothesis is conditional. A learned prior of building layouts helps, but only when training distribution matches deployment, the step budget is tight, and inference is fast enough to drive. The rest of the talk tests each one."**

`[Scroll into Section 2]`

---

## SECTION 2 — OUR SYSTEM  (0:55 → 1:30)

### Card: Flow diagram (5 colored steps)

`[The 5-step flow diagram is on screen]`

> **"The system is one loop, five steps. Build a partial map from accumulated lidar. Run the U-Net with DDIM to imagine K equals 8 plausible full buildings. Score every candidate frontier under those 8 imagined buildings. Drive to the argmax. Then repeat."**

### Card: 4-card pipeline grid (01 Data, 02 Model, 03 Scoring, 04 Integration)

`[Scroll to the 4-card grid below the flow]`

> **"Four stages in the broader pipeline. Data: HouseExpo, 35 thousand floor plans, 2.66 million training pairs after augmentation. Model: a conditional U-Net with 4.16 million parameters. Scoring: K equals 8 DDIM samples per decision, score is expected gain plus variance bonus minus distance. Integration: a ROS 2 node publishes to slash best frontier and the exploration manager from PA3 drives the robot."**

### Card: Data (metrics + sample grid image)

`[Scroll to the Data card with the 20 training pairs]`

> **"This is what the training data looks like. For each pair, the ground truth occupancy grid on the left, the partial map the robot would actually see in the middle, and the hidden remainder we ask the model to predict on the right."**

### Card: Model (table)

`[Scroll to the Model table]`

> **"Standard conditional U-Net. 4 million parameters. Three input channels: the noisy target, the partial map, and a known-mask. Trained with DDPM, 1000 noise timesteps. At inference I use DDIM with 30 steps to make it 30 times faster."**

`[Scroll into Section 3]`

---

## SECTION 3 — THE MATH  (1:30 → 2:05)

### Card: How diffusion works (cat image)

`[The diffusion explainer cat image is on screen]`

> **"Quick visual on how diffusion works. You train by gradually corrupting clean data with noise. At inference, you start from pure noise and run the model backward to generate a fresh sample. We do exactly that, except the data is building floor plans instead of cat pictures."**

### Card: DDPM forward process (formula)

`[Scroll to the first formula]`

> **"This is the standard DDPM forward process. Clean data plus scaled Gaussian noise. Alpha bar is the cumulative noise schedule. At time zero we have the clean image, at time T equals 1000 we have pure noise."**

### Card: Training objective (formula)

`[Scroll to the second formula]`

> **"The training loss is MSE between the noise the model predicts and the real noise that was added. The U-Net is conditioned on the partial map and the known mask, so it knows which parts of the image are real input and which it needs to imagine."**

### Card: Frontier scoring (plain-English line + formula + scoring image)

`[Scroll to the third formula]`

> **"This is the new logic. For each candidate frontier, we average the expected information gain across the 8 imagined buildings, add a variance bonus when those 8 disagree, subtract distance, and pick the highest score."**

> **"The variance term is the upper confidence bound trick from bandits. When the 8 samples disagree about what is behind a door, the model is telling us it does not know what is there, so we push the robot toward that frontier to find out."**

`[Scroll into Section 4]`

---

## SECTION 4 — TRAINING  (2:05 → 2:30)

### Card: Loss convergence (loss curves + metrics)

`[Loss convergence is on screen]`

> **"29 epochs on a T4 GPU. The MSE loss on the noise dropped two orders of magnitude, from 0.033 down to 0.0007. Validation IoU averaged 0.66 with the best epoch hitting 0.77."**

### Card: Sample quality across epochs

`[Scroll to sample progression panel]`

> **"This shows the model waking up across training. Epoch 5 is blurry blobs. Epoch 20 has crisp walls and rooms. I deployed the epoch 20 checkpoint."**

### Card: K = 8 imagined buildings (diversity image)

`[Scroll to K=8 diversity grid]`

> **"Same partial input on the left, eight different DDIM samples on the right. Where the partial map already constrains structure, the samples agree. Where it does not, they diverge. That disagreement is exactly the exploration signal."**

### Card: Denoising GIF

`[Scroll to denoising animation]`

> **"And here is what it looks like animated. Four samples starting from pure Gaussian noise, denoising in lockstep. By the end they have settled into four distinct plausible buildings."**

`[Scroll into Section 5]`

---

## SECTION 5 — DEMOS: THE LOOP RUNNING  (2:30 → 3:45)

### Demo 1: The headline animation  (15s)

`[The diffusion exploration GIF is on screen]`

> **"This is the loop running end to end. Left panel is the partial map with scored frontiers. Middle panel is the model's prediction sharpening as scans accumulate. Right panel is the ground truth, which the robot does not see. Far right panel is coverage climbing to 91 percent in 11 steps."**

### Demo 2: Diffusion vs baseline  (10s)

`[Scroll to side-by-side GIF]`

> **"Same map, two robots. Top row is our diffusion scorer. Bottom row is the classical heuristic. Same start position, same lidar. Watch the top row reach high coverage one or two steps before the bottom."**

### Demo 3: Four scorers synchronized  (25s)

`[Scroll to 4-method synchronized GIF]`

> **"This is what the professor specifically asked for. All four frontier scorers running on the same map at the same time, with identical start position and seed."**

> **"Top left is nearest, which just picks the closest unknown. Top right is info-gain only, which picks where the most cells are unknown. Bottom left is info-gain plus distance, which is the textbook scorer that balances both terms. Bottom right is ours, with K equals 8 imagined buildings."**

> **"Watch the coverage tickers. Nearest crawls. The info-gain variants commit to a wing. Diffusion bottom right reaches 95 percent by step 4."**

### Demo 4: Behind the mind  (15s)

`[Scroll to behind-the-mind composite]`

> **"Three rows per step. Top is the world the robot does not see, with its accumulated trail. Middle is the partial map plus frontier candidates. Bottom is the mean of the imagined buildings with the picked frontier circled. By step 4 the imagination already matches reality."**

### Demo 5: Stage simulator  (10s)

`[Scroll to Stage maze GIF]`

> **"And here is the diffusion scorer driving an actual ROS 2 robot in the Stage maze world. Real nodes on the slash map and slash best frontier topics. Proof the pipeline runs end to end."**

`[Scroll into Section 6]`

---

## SECTION 6 — RESULTS  (3:45 → 4:45)

### Chart 1: Four scorers, 30 maps, same seeds  (25s)

`[The 4-baseline curves chart is on screen]`

> **"X-axis is the frontier decision step. Y-axis is mean coverage across 30 held-out HouseExpo maps. Each curve is one of the four scorers."**

`[POINT AT step 4]`

> **"At step 4, the decomposition is clean. Going from nearest, the grey curve, to info-gain alone, the purple curve, is worth plus 17 points. Info-gain is doing most of the work. Adding the distance penalty, the blue curve, contributes basically nothing, the curves overlap. The diffusion prior, the pink curve, adds another 3.5 points on top."**

> **"In one line: info-gain matters, distance is decorative, the prior buys a real but small early-budget head start."**

### Chart 2: Out-of-domain ablation  (20s)

`[Scroll to OOD chart]`

> **"This is the kill experiment. I trained a second U-Net from scratch on synthetic warehouse layouts, then evaluated it on the same 30 residential HouseExpo maps. Wrong-domain prior, same evaluation."**

> **"The in-domain delta bumps to plus 3.5 points at step 4. The out-of-domain delta sits flat at zero. That is the cleanest possible proof that the prior is doing real structural work, not just averaging."**

### Chart 3: Realtime upper bound  (15s)

`[Scroll to 2x2 matrix]`

> **"Current inference is 2 seconds per decision. Too slow for a real robot. So I asked: what if it were realtime? Re-sample the 8 imagined buildings every 3 driving cells, the speed a distilled model could achieve."**

> **"In-domain plus realtime gets to plus 10 points. Sampling frequency contributes more than prior match in our data. Distillation is the path."**

`[Scroll into Section 7]`

---

## SECTION 7 — HARDSHIPS  (4:45 → 5:00)

`[Hardships bullets are on screen]`

> **"Quickly, what did not work. Docker QEMU on Apple Silicon is 50 times too slow for live diffusion in Stage, so the Stage integration uses pre-computed waypoints. The baseline catches up by step 20. Out of domain priors give zero help. Two of 30 maps are hard failures. Live hardware is blocked by the same latency wall."**

`[Scroll into Section 8]`

---

## SECTION 8 — WHAT IS NEXT  (5:00 → 5:10)

`[What is next bullets are on screen]`

> **"Four next steps. Distillation to a Jetson-sized model for sub-second inference. Richer training domains like Gibson and Matterport. K-sample disagreement as an I-do-not-know flag. Live ROSbot deployment in a matched-distribution building."**

---

## CLOSE  (5:10 → 5:15)

> **"That is the project. Thank you. Happy to take questions."**

---

## Timing table

| Section | What is on screen | Time | Cumulative |
|---|---|---|---|
| Hero | Title + name | 0:10 | 0:10 |
| Inspiration | Related work + hypothesis | 0:45 | 0:55 |
| Our System | Flow + 4-cards + data + model | 0:35 | 1:30 |
| Math | Cat + 3 formulas | 0:35 | 2:05 |
| Training | Loss + samples + diversity + denoising GIF | 0:25 | 2:30 |
| Demos | 5 demos in order | 1:15 | 3:45 |
| Results | 3 charts | 1:00 | 4:45 |
| Hardships | Bullets | 0:15 | 5:00 |
| What is next | Bullets | 0:10 | 5:10 |
| Close | Questions? | 0:05 | 5:15 |

**Total 5:15** against the 6:00 hard cap. About 45 seconds of slack.

If you need to cut, the easiest 30 seconds to drop:
- One of the Math formula descriptions (just point and say "and this is the training objective")
- One of the smaller demos (drop Demo 4 Behind the Mind if running long)

---

## If a GIF does not load

Stay calm. Describe it in one sentence and move on:

> "The animation is not loading. What you would see is the partial map on the left, the model's prediction in the middle, ground truth on the right, and coverage climbing. Moving on."

## If the laptop crashes

Recite this paragraph from memory:

> "I trained a conditional diffusion U-Net on 35 thousand HouseExpo floor plans. At each frontier decision the robot samples K equals 8 imagined completions and scores each frontier by mean expected information gain plus variance bonus minus distance. On 30 held-out maps this gives a 3.5 point early-budget advantage at step 4. The out-of-domain ablation drops it to zero, which proves the prior is doing real structural work. If inference were sub-second, the advantage extrapolates to 10 points. Stage integration runs via pre-computed waypoints because Docker QEMU on Apple Silicon is too slow for live diffusion."

---

## Q&A prep

### "What are your 4 baselines exactly?"
> "Nearest: argmin distance. Info-gain only: argmax unknown cells. Info-gain plus distance: the textbook scorer. Diffusion: same as the textbook scorer but the gain is averaged over 8 imagined buildings with a variance bonus."

### "Why diffusion instead of a single regression?"
> "A regression network would average all plausible completions into a blurry mush. Diffusion gives me distinct crisp samples, and the variance across them is the exploration signal a regressor cannot provide."

### "Why pre-computed waypoints in Stage?"
> "Docker QEMU on Apple Silicon adds a 50x latency penalty. With diffusion already at 2 seconds per decision on T4, running it inside emulated Stage would be 100 seconds per frontier. On real T4 hardware K=4 with DDIM 10 drops to 500 milliseconds, which is the realtime budget."

### "How is this different from Diffusion Forcing or full world models?"
> "Diffusion Forcing is the more ambitious version, sequential prediction over multiple exploration timesteps. What I built is the single-step version, listed as future work."

### "What is the ethics dilemma?"
> "Dual use. Same prior that helps search and rescue could let a surveillance robot infer a private floor plan. Mitigations: opt-in deployment, on-device redaction of sensitive rooms, broader training data audits. Full discussion is in the report."

---

**Print or open on phone. Walk through it once tonight aloud. Time yourself. You are ready.**
