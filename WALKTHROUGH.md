# Walkthrough script

Open this on your phone. Read top to bottom. Total about 5 minutes.

---

## The four scorers I compare (memorize this)

The professor asked for a richer comparison than just diffusion vs the conventional baseline. So I implemented four scorers and ran all of them on the same maps with the same seeds.

| # | Scorer | What it does | One-liner to say |
|---|---|---|---|
| 1 | **Nearest** | Picks the closest frontier. Ignores how much you would learn there. | "Just go to the nearest unknown spot." |
| 2 | **Info-gain only** | Picks the frontier that uncovers the most unknown cells. Ignores distance. | "Go to wherever you would learn the most." |
| 3 | **Info-gain + distance** | The textbook heuristic. Info-gain minus a distance penalty. | "Learn the most while not walking too far." |
| 4 | **Diffusion (ours)** | For each frontier, score it under 8 imagined buildings. Use mean gain + variance bonus - distance. | "Learn the most, especially in places the model is unsure about, without walking too far." |

**The clean decomposition I found:**
- Going from nearest (1) to info-gain (2): **+17.5 points** at step 4. Info-gain is doing most of the work.
- Adding the distance penalty (2 to 3): **basically zero.** Distance is decorative.
- Adding the diffusion prior (3 to 4): **+3.5 points**. Real but small early-budget head start.

---

## Walkthrough by page section

### Open (10 seconds)
"Hi, I am Moin. My project is Diffusion-Guided Frontier Exploration. I trained a diffusion model on real building floor plans and used it as a structural prior to help a robot decide where to explore next."

### Section 0 — Inspiration (40 seconds)
"Think about walking into a friend's house you have never been to. You see a leash by the door and a chewed toy in the corner, and you already predict there is probably dog food in the kitchen. You do not just see what is there, you imagine what is behind the rooms you have not entered. That is the world model intuition.

The biggest names in AI are building this for production robotics right now. NVIDIA Cosmos trained world foundation models on 20 million hours of robot and driving video. DeepMind Genie 2 generates interactive 3D worlds. Decart shows real time generative world models on commodity GPUs are now possible.

My project is the simplest possible instance for a class robotics problem. The hypothesis is conditional: a learned prior helps only when training matches deployment, the step budget is tight, and inference is fast enough to drive."

### Section 2 — System (35 seconds)
"Four stages. First, the data: HouseExpo, 35 thousand floor plans, 2.66 million training pairs after augmentation. Second, the model: a conditional U-Net with 4.16 million parameters, standard DDPM. Third, the scoring: K equals 8 DDIM completions per decision, score is expected gain plus variance bonus minus distance. Fourth, the integration: a ROS 2 node publishes to /best_frontier and the exploration manager from PA3 drives the robot."

### Section 3 — Math (35 seconds)
Point at the cat diagram.
"This is how diffusion works. Train by corrupting clean data with noise. At inference, start from pure noise and reverse the process to generate a fresh sample."

Point at the DDPM formula.
"Standard forward process. Clean data plus scaled Gaussian noise. We train the U-Net to predict the noise."

Point at the scoring formula.
"For each frontier we average expected info gain across 8 imagined buildings, add a bonus when they disagree, subtract distance, pick the highest."

### Section 4 — Training (25 seconds)
"29 epochs on a T4. Loss dropped two orders of magnitude. Validation IoU averaged 0.66, best 0.77.

The progression panel shows the model waking up. Epoch 5 is blurry blobs, epoch 20 has crisp walls.

The denoising GIF shows 4 samples emerging from pure noise into 4 distinct plausible buildings. That disagreement is exactly the exploration signal we use."

### Section 5 — Demos (60 seconds)

**Demo 1 — headline animation:** "Left: partial map with scored frontiers. Middle: model's prediction. Right: ground truth. Far right: coverage. 91 percent in 11 steps."

**Demo 2 — side by side:** "Top is diffusion. Bottom is baseline. Same map, same start. Watch the top reach high coverage first."

**Demo 3 — four scorers synchronized:** "All four methods on the same map at the same time. Top left nearest, top right info-gain only, bottom left info-gain plus distance, bottom right diffusion. Watch the coverage tickers. Diffusion bottom right reaches 95 percent by step 4."

**Demo 4 — behind the mind:** "Three rows: world, robot's partial view, model's imagined building. By step 4 the imagination matches reality."

**Demo 5 — Stage simulator:** "The diffusion scorer driving a ROS 2 robot in the PA3 maze world. Real ROS 2 nodes on the /map and /best_frontier topics."

### Section 6 — Results (60 seconds)

**Chart 1 — four scorers, 30 maps:**
"X-axis is the frontier decision step. Y-axis is mean coverage across 30 maps. Pink is diffusion, blue and purple are info-gain variants, grey is nearest. Diffusion leads in the early steps. The two info-gain curves overlap, meaning distance contributes nothing. All converge by step 10."

**Chart 2 — out-of-domain ablation:**
"This is the kill experiment. I trained a second U-Net on synthetic warehouses and evaluated it on residential HouseExpo. With the matching prior the advantage bumps to +3.5 points. With a mismatched prior it sits at zero. Proves the prior is doing real structural work, not averaging."

**Chart 3 — realtime upper bound:**
"2 by 2 of prior type and sampling mode. In-domain plus discrete is the baseline +3.5pp. In-domain plus realtime is +10pp. OOD plus discrete is the kill at zero. OOD plus realtime claws back +6pp. Sampling frequency matters more than prior match. Distillation is the path."

### Section 7 — Hardships (20 seconds)
"Docker QEMU on Apple Silicon is 50 times too slow for live diffusion in Stage, so the Stage integration uses pre-computed waypoints. The baseline catches up by step 20. OOD priors give zero help. Two of 30 maps are hard failures. Live hardware blocked by the same latency wall."

### Section 8 — What is next (15 seconds)
"Distillation to a Jetson-sized model for sub-second inference. Richer training domains, Gibson and Matterport. K-sample disagreement as an I-do-not-know flag. Live ROSbot deployment in a matched-distribution building."

### Close (5 seconds)
"That is the project. Thank you. Happy to take questions."

---

## Timing

| Section | Allowance | Cumulative |
|---|---|---|
| Open | 0:10 | 0:10 |
| Inspiration | 0:40 | 0:50 |
| System | 0:35 | 1:25 |
| Math | 0:35 | 2:00 |
| Training | 0:25 | 2:25 |
| Demos | 1:00 | 3:25 |
| Results | 1:00 | 4:25 |
| Hardships | 0:20 | 4:45 |
| What is next | 0:15 | 5:00 |
| Close | 0:05 | 5:05 |

About 5 minutes. One minute of slack against the 6 minute hard cap.

---

## If the prof asks about the 4 baselines specifically

Open the 4-scorers synchronized GIF in Demos, or the 4-baseline curves chart in Results, and say:

> "All four scorers are running on the same map at the same time. Nearest just goes to the closest unknown spot. Info-gain only goes to wherever the most cells are unknown. Info-gain plus distance is the textbook scorer that balances both. Diffusion does the same thing but averages over 8 imagined buildings, with a variance bonus for spots the model is unsure about. The decomposition is clean: info-gain is doing most of the work, distance contributes nothing, the diffusion prior adds another 3.5 points on top in the early budget."
