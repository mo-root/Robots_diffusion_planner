# Learned Map-Completion Priors for Mobile-Robot Frontier Exploration

**Author:** Moin Mattar
**Course:** COSC 81/281 · Spring 2026 · Final Project Proposal (FP2)
**Date:** 2026-05-20
**Format:** Solo · ROS 2 Humble · Stage simulator · builds directly on PA3 + PA4
**Track:** Frontier Explorer (per the spec's example tracks)

---

## Abstract

Frontier-based exploration scores each candidate frontier with a hand-crafted heuristic that is blind to the structure of the unseen environment. We propose to learn a generative prior over completed occupancy grids using a small **conditional denoising diffusion model**, and use samples from that prior to score frontiers by *expected* information gain with an *uncertainty bonus*. The method is trained on synthetic (partial-map, full-map) pairs from the **HouseExpo** dataset and evaluated in Stage on the maze world from PA3 plus additional procedurally generated worlds. We hypothesize that sampling-based scoring reduces time-to-coverage relative to standard frontier exploration, with the largest gains in environments with structural ambiguity (corridor vs. dead-end). We discuss extensions to **Diffusion Forcing** for variable-horizon rollouts as future work.

---

## 1. Introduction

Standard frontier-based exploration (Yamauchi 1997; Quin et al. 2014, the algorithm covered in Lec 19) picks the next frontier with a hand-crafted score, typically:

$$\text{score}(f) = \alpha \cdot N_\text{unknown}(f) - \beta \cdot d(r, f)$$

where $N_\text{unknown}(f)$ is the number of unknown cells visible from frontier $f$ and $d(r,f)$ is the robot's distance to it. This heuristic ignores **what lies beyond the frontier.** A frontier opening into a wide unmapped room and a frontier opening into a dead-end closet can score identically; only after the robot drives there does the difference manifest.

Learned map-completion priors address this. Shrestha et al. (ICRA 2019) trained a conditional GAN to predict the full map from a partial observation and used the predicted free-space to inform frontier scoring. More recent work uses diffusion models (Lin et al. 2024) for 3D occupancy prediction at the frontier. Both these methods produce a **single deterministic prediction** and use it as a point estimate of "what's behind the wall."

Our contribution is to treat the unseen environment as **inherently uncertain** and score frontiers by the **distribution** of plausible completions:

> For each candidate frontier $f$, sample $K$ completions from the diffusion model. Score $f$ by $\mathbb{E}[\text{info gain}] + \lambda \cdot \text{Var}[\text{info gain}]$.

The variance bonus encourages the robot toward frontiers where the model is *uncertain*, which are the frontiers where new observations are most informative.

---

## 2. Method

### 2.1 Map Completion Model

**Architecture.** A small conditional U-Net (~5M parameters) operating on 256×256 occupancy grid patches. Inputs: a partial grid where each cell is one of $\{$free, occupied, unknown$\}$ (encoded as a 3-channel one-hot image). Output: a predicted full grid (3 channels, softmaxed per cell).

**Training objective.** Standard conditional DDPM (Ho et al. 2020):
$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t, c) \|^2\right]$$
where $c$ is the partial map (conditioning) and $x_0$ is the ground-truth full map.

**Training data.** **HouseExpo** dataset (Li et al. IROS 2020): 35,000+ indoor floor plans. For each map, we generate (partial, full) pairs by simulating a partial-observation mask:
- Place a virtual robot at a random cell
- Cast a 360° lidar with realistic range and angular resolution (matching the Stage rosbot's lidar)
- Mark observed cells as known, the rest as unknown
- The resulting partial map is the conditioning $c$, the original full map is $x_0$

This yields ~500k training samples cheaply.

### 2.2 Sampling-Based Frontier Scoring

For each candidate frontier $f$ at inference time:
1. Crop the current partial map to a 256×256 window centered on the robot
2. Sample $K = 8$ completions from the diffusion model conditioned on the partial map
3. For each completion, compute the realized info gain: the number of currently-unknown cells that the completion predicts as **free and reachable from $f$** (within a configurable radius)
4. Score $f$ by:
$$\text{score}(f) = \mathbb{E}_k[g_k(f)] + \lambda \cdot \text{Std}_k[g_k(f)] - \beta \cdot d(r, f)$$
where $g_k(f)$ is the info gain under completion $k$, $\lambda$ controls the uncertainty bonus, and the last term is the standard distance penalty.
5. Pick $f^* = \arg\max_f \text{score}(f)$, drive to it, scan, update the partial map, repeat.

### 2.3 ROS 2 Integration

The system is one ROS 2 node that subscribes to `/map` (from a slightly modified PA4 mapper that also tracks frontier cells) and publishes both `/predicted_map` (the mean completion, for visualization) and `/scored_frontiers` (geometry_msgs/PoseArray, each pose annotated with its score). A simple pure-pursuit controller drives the robot to the highest-scoring frontier and the loop continues.

---

## 3. Connection to Course Material

| Class component | Where in this project |
|---|---|
| **Lec 10** — occupancy grids | The map representation; `pa4_grid_mapper.py` is the substrate |
| **Lec 12** — search-based planning | A* used to compute distance $d(r, f)$ and reachability inside each completion |
| **Lec 14-16** — Bayes filtering | The PA4 log-odds update; the diffusion model is a learned generative version of the same posterior |
| **Lec 19** — exploration & coverage | The frontier-based exploration loop is direct from lecture (Quin et al. 2014) |
| **Lec 20** — ML for robotics | The conditional diffusion model is the ML component |

The project synthesizes **five distinct lecture units** into one autonomous system, as the rubric requires.

---

## 4. Experimental Plan

### 4.1 Worlds

- **PA3 maze** (small, structured) — already in the workspace
- **Two procedurally generated mazes** (one with dead-ends, one with one large central room)
- **Two HouseExpo floorplans** loaded into Stage (out-of-training-distribution)

Five worlds × ten trials each = 50 trials per condition.

### 4.2 Baselines

- **Heuristic frontier.** Standard $\text{score}(f) = N_\text{unknown}(f) - \beta d(r, f)$.
- **Shrestha-style.** A deterministic conditional U-Net (no diffusion) trained on the same data, single prediction used for info gain. This isolates "stochastic sampling" from "learned prior."
- **Ours (no variance bonus).** $\lambda = 0$. Isolates "expected info gain" from "uncertainty-seeking."
- **Ours (full).** $\lambda > 0$.

### 4.3 Metrics

- **Time-to-90%-coverage** (primary): wall-clock seconds in sim to reach 90% of free cells observed.
- **Path length** to 90% coverage.
- **Info-gain calibration**: reliability diagram of predicted vs. realized info gain.
- **Qualitative**: side-by-side visualization of $K=8$ completions per frontier (the headline demo figure).

### 4.4 Ablations

- $K \in \{1, 4, 8, 16\}$ — does more samples help?
- $\lambda \in \{0, 0.5, 1.0, 2.0\}$ — variance-bonus sweep
- Training data scale: subsample HouseExpo to 10%, 50%, 100% and check generalization

---

## 5. Risks and Fallbacks

| Risk | Mitigation |
|---|---|
| Diffusion training won't converge in time | Diffusion U-Nets are well-trodden territory. There are dozens of working open-source implementations (`huggingface/diffusers`, `lucidrains/denoising-diffusion-pytorch`). If training fails, swap for a deterministic U-Net regressor — degrades to the Shrestha-style baseline (still novel: sampling becomes dropout-based) |
| Sim-to-sim domain gap (HouseExpo → Stage) | Include 2 Stage worlds in training. Domain-randomization on lidar noise. |
| ROS 2 integration eats time | The mapper, frontier extractor, and pure-pursuit controller are already implemented in PA3/PA4. Only the diffusion-inference module is new. |
| Inference too slow for real-time | At 256×256 and $K=8$, denoising takes ~1-2s on a laptop GPU. Acceptable because frontier selection happens once every ~5 seconds during driving. |

---

## 6. Future Work — Diffusion Forcing for Variable-Horizon Rollouts

The current method predicts a *single-shot* completion. A natural extension is to predict the **evolution of the map over time** as the robot moves toward a candidate frontier — a video-generation problem rather than image-completion. Diffusion Forcing (Chen et al. NeurIPS 2024) is purpose-built for this: it trains a sequence model with **independent per-token noise levels**, enabling **variable-horizon rollouts** with calibrated per-step uncertainty.

If time permits past FP3 (June 1), I will adapt the open-source `buoyancy99/diffusion-forcing` repo to roll out (map_patch, action) sequences and score frontiers by expected info gain over a 20-step horizon. This is flagged as **stretch work, not a deliverable.**

---

## 7. Timeline

| Date | Milestone |
|---|---|
| **May 20-22** | Repo setup. Reproduce a HuggingFace diffusion tutorial on MNIST to confirm pipeline. |
| **May 22-26** | Data generation: HouseExpo loader, partial-map simulator, ~500k training samples cached. |
| **May 26-27** | First training run begins. **FP2.5 in-class progress update (May 27): show loss curves + sample completions.** |
| **May 27-30** | Training converges. Implement frontier scorer. Integrate into PA4 ROS node. |
| **May 30-Jun 1** | Run baselines + ablations in Stage. Generate plots. **FP3 (June 1): 6-min presentation + demo video.** |
| **Jun 1-Jun 6** | Write 4-page IEEE paper. Polish demo. Polish github repo. **FP4 (June 6): final report + code + 3-min teaser.** |

---

## 8. Deliverables Summary

- 4-page IEEE-format final report
- Open-source github repo with training + inference + ROS 2 integration
- Demo video showing the system in three Stage worlds
- Quantitative comparison against 3 baselines on 5 worlds × 10 trials

---

## 9. References

- Yamauchi, B. *A frontier-based approach for autonomous exploration.* CIRA 1997.
- Quin, P. et al. *Frontier-based multi-robot exploration with rendezvous.* 2014 (Lec 19).
- Shrestha, R. et al. *Learned map prediction for enhanced mobile robot exploration.* ICRA 2019.
- Ho, J., Jain, A., Abbeel, P. *Denoising diffusion probabilistic models.* NeurIPS 2020.
- Li, T. et al. *HouseExpo: a large-scale 2D indoor layout dataset for learning-based algorithms on mobile robots.* IROS 2020.
- Lin et al. *Online diffusion-based 3D occupancy prediction at the frontier with probabilistic map reconciliation.* 2024. arXiv:2409.10681.
- Chen, B. et al. *Diffusion forcing: next-token prediction meets full-sequence diffusion.* NeurIPS 2024. arXiv:2407.01392.

---

## Talking Points for the Meeting

> **Goal of the meeting:** confirm the prof is happy with this scope.

1. **One-sentence summary.** "I'm going to learn a generative prior over completed occupancy grids using a small conditional diffusion model, and use *samples* from that prior to score frontiers by *expected* information gain plus an uncertainty bonus."

2. **Why this scope (not the bigger diffusion-forcing version).** "Diffusion Forcing is genuinely interesting but adds complexity I don't think I can safely deliver in three weeks solo. I'd rather ship a clean, well-evaluated single-shot version and flag DF as future work — that way the contribution is real, not aspirational."

3. **Why this is novel over Shrestha 2019 / the 2024 frontier paper.** "Both of those methods predict a *single* map and use it as a point estimate. I'm sampling multiple completions and scoring frontiers by expected gain + variance. The variance term is the key — it lets the robot explore in a way that's aware of its own uncertainty."

4. **Class connection.** "PA4 is the substrate. PA3 gives me the planner. Lec 19 gives me the exploration loop. Lec 20 gives me the ML method. Five units stitched together — that's the synthesis the rubric asks for."

5. **What I want from the prof.** Ask:
   - Does he buy the novelty claim (sampling-based scoring with variance bonus over Shrestha's deterministic version)?
   - Is HouseExpo + Stage the right experimental setup, or does he want Gazebo?
   - Should I plan to run on real hardware (ROSbot) at the end, or is Stage enough?
   - Is "future work: Diffusion Forcing" a good framing, or is he expecting me to actually deliver it?

6. **Honest about risk.** "Worst case, training fails and the diffusion model degrades to a deterministic regressor. That's still novel (sampling via dropout instead of diffusion) and I keep all the rest of the system. So I have a guaranteed minimum deliverable."
