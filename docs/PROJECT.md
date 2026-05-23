# Final Project: Diffusion-Based Map Completion for Frontier Exploration

## What is this project about (plain English)

You have a robot exploring an unknown building. At any point, it has a **partial map**: some cells are known (free or wall), the rest are unknown (gray fog). The robot needs to decide **which direction to explore next**.

Standard frontier exploration uses a dumb heuristic: "go toward the biggest patch of unknown." It has no idea whether that unknown patch hides a big open room or a dead-end closet.

**Your project:** train a small neural network (conditional diffusion model) that looks at the partial map and **predicts what the full map probably looks like**. Then use those predictions to make smarter frontier choices.

Think of it like this:

```
Current state:          What the model predicts:       What the robot decides:

 ████████████            ████████████                  "Frontier B probably leads
 █·····██···█            █·····██···█                   to a big room. Go there
 █·····██···█            █·····██···█                   instead of Frontier A
 █··R··██···█    --->    █··R··██···█    --->           (which is probably a
 █·····??????            █·····█····█                    dead-end closet)."
 █·····??????            █·····█    █
 ████████████            ████████████

 · = free  █ = wall  ? = unknown  R = robot
```

The diffusion model can also generate **multiple different predictions** (because the unseen area is genuinely uncertain). So you sample several completions and score each frontier by the average predicted info gain.

## What you build (3 phases, per the prof's advice)

### Phase 1: Train the model (standalone, no simulator needed)

**Input:** A partial occupancy grid (some cells known, rest masked as unknown).
**Output:** A predicted full occupancy grid.

**Training data:** Take real floor plans from the HouseExpo dataset (35,000+ indoor layouts, free download). For each floor plan:
1. Pick a random robot position
2. Simulate what the robot's lidar would see from that position (which cells are visible)
3. Mark everything else as "unknown"
4. Now you have a (partial map, full map) pair

Generate ~500k of these pairs. The model learns: "given this partial view, what does the rest of the building probably look like?"

**The model:** A conditional U-Net (~5M parameters). This is the standard architecture for image-to-image tasks. It takes the partial map as input (conditioning) and learns to denoise a noisy version of the full map. At inference, you start from pure noise and iteratively denoise to get a predicted full map.

**Data augmentation (prof suggested this):** Rotate and flip the same map to get 4x-8x more training samples. This helps the model generalize to different orientations.

**What success looks like:** The predicted maps look structurally plausible. Walls continue in reasonable directions. Rooms have realistic shapes. You can measure this with IoU (intersection over union) between predicted and actual maps on a held-out test set.

### Phase 2: Integrate with your robot in Stage simulator

Once the model is trained and predicting reasonable maps:

1. Your PA4 mapper (`pa4_grid_mapper.py`) builds the partial map from lidar scans in real time
2. A new ROS 2 node subscribes to `/map`, crops a window around the robot, and feeds it to the trained diffusion model
3. The model outputs K=8 predicted completions
4. For each frontier cell on the boundary of known/unknown, score it:
   - For each of the 8 completions, count how many new free cells you'd discover by going to that frontier
   - Average those counts = expected info gain
   - Also compute the variance (how uncertain the model is about this frontier)
   - Final score = expected gain + bonus for high variance (explore uncertain areas)
5. Pick the best frontier, drive to it (pure-pursuit, same as PA3), repeat

**What success looks like:** The robot explores a Stage world and makes smarter choices than the baseline (which uses the standard "biggest unknown patch" heuristic). Even if it's only slightly better, the fact that the diffusion model is producing meaningful predictions is the contribution.

### Phase 3 (stretch): Deploy on real ROSbot

If Phase 2 works well in Stage, try it on the real robot in a lab hallway. The prof mentioned this as aspirational, not required.

## How this connects to everything you already did in class

| What you already built | How it's reused |
|---|---|
| **PA4 occupancy grid mapper** (Bresenham + log-odds) | Produces the `/map` topic that feeds the diffusion model. Your mapper IS the perception layer. |
| **PA3 path planner** (A* + pure-pursuit follower) | Drives the robot to the chosen frontier. Your planner IS the action layer. |
| **Lec 19 frontier exploration** (Quin et al. 2014) | The exploration loop: detect frontiers, score them, pick one, go. You're replacing the scoring function. |
| **Lec 14-16 Bayes filtering** | The log-odds update in PA4. The diffusion model is a learned generative version of the same idea: given observations, what's the posterior over maps? |
| **Lec 20 ML for robotics** | The diffusion model itself. |

## Why diffusion instead of just a regular neural network?

A regular neural network (plain U-Net regressor) would predict **one** completed map. Diffusion gives you two advantages:

1. **Multiple samples.** Run the model 8 times with different random seeds and you get 8 different plausible completions. Some might predict a room behind the wall, others might predict a corridor. This diversity is real information: it tells you how uncertain the model is about each area.

2. **Better image quality.** Diffusion models produce sharper, more coherent images than regressors (which tend to output blurry averages). For maps, "sharp" means clean wall boundaries and realistic room shapes, which matters for info-gain scoring.

## What the prof specifically said matters for grading

From the meeting:

> "Even if it's not completely achieving the full task, if you can show that from the beginning up to now it's improving, that's OK too."

Translation: **you don't need to beat baselines**. Showing the model learns (loss goes down, predictions get better over training) is a passing outcome. Beating the baseline is a strong outcome.

> "The initial part can be completely separate from the simulator."

Translation: **Phase 1 (training) is standalone**. You can make progress without touching ROS or Stage at all. Just train on HouseExpo images.

> "Think about how to generate the corresponding data so that you can train the model reasonably well."

Translation: **data pipeline is the first critical step**. Get HouseExpo loading, partial-map simulation, and augmentation working before touching the model.

> "Maybe you can augment the data. For example, for the same map you can have different orientations."

Translation: **rotation augmentation is expected**. Rotate maps by 90/180/270 degrees. Flip horizontally/vertically. Cheap, effective, professor-recommended.

## Baselines for comparison

1. **Heuristic frontier** (no learning): standard score = unknown cells visible minus distance cost. This is what lec 19 teaches.
2. **Deterministic U-Net** (learning, but no diffusion): same architecture, but trained as a regressor (MSE loss) instead of diffusion. Single prediction. This isolates "does diffusion's multi-sample capability actually help?"
3. **Ours (diffusion, no variance bonus)**: expected gain only, no uncertainty bonus. Isolates whether the variance term matters.
4. **Ours (full)**: expected gain + variance bonus.

## Diffusion Forcing (future work, NOT the current project)

Diffusion Forcing is a training technique (not a model) that lets you predict sequences with independent noise levels per step. It would let you predict **how the map evolves over multiple timesteps** (instead of a single-shot completion). We discussed it as interesting but too complex for the current timeline. It stays as a "Future Work" section in the paper.

## Timeline

| When | What | Depends on simulator? |
|---|---|---|
| **May 20-22** | Download HouseExpo. Write data loader. Generate first batch of (partial, full) pairs. | No |
| **May 22-26** | Build U-Net. Start training. Monitor loss curves. | No |
| **May 27** | **Wednesday progress check-in.** Show: loss curves, sample predictions, "here's what the model sees vs. what it predicts." | No |
| **May 27-30** | Finish training. Write frontier scorer node. Integrate with PA4 mapper in Stage. | Yes (Stage) |
| **May 30-Jun 1** | Run baseline vs. ours in 3-5 Stage worlds. Generate plots. | Yes |
| **June 1** | **Final presentation (6 min).** Show demo video + results. | Done |
| **June 1-6** | Write 4-page IEEE report. Polish demo video. Clean up github repo. | Done |
| **June 6** | **FP4: final report + code + 3-min teaser video.** | Done |

## What to show at the Wednesday check-in (May 27)

Minimum:
- "Here's a partial map. Here's what my model predicts. Here are 4 different samples showing the model's uncertainty."
- Loss curve showing improvement over training epochs.
- Brief slide on how you'll integrate with Stage (even if not done yet).

That's enough. The prof said this check-in is casual.

## Deliverables summary

1. 4-page IEEE paper
2. Github repo (data pipeline + training + inference + ROS 2 node)
3. 3-min demo video showing the robot exploring with the diffusion prior in Stage
4. Quantitative comparison: time-to-90%-coverage, ours vs. baselines, on 3-5 worlds

## Risk and what to do if things go wrong

| Problem | Solution |
|---|---|
| Diffusion model doesn't converge | Switch to a plain U-Net regressor (MSE loss). Everything else stays the same. You lose the multi-sample capability but keep the learned prior. |
| Predictions are blurry or bad | Add more training data (synthetic = cheap). Try more augmentation. Increase model capacity (add layers). Tune learning rate. Prof specifically said this is a solvable problem. |
| Stage integration breaks | Show results on offline evaluation only (run predictions on held-out HouseExpo test maps, measure IoU). The prof said "the initial part can be separate from the simulator." |
| Not enough time for baselines | At minimum compare against the heuristic frontier baseline. The deterministic U-Net baseline is nice-to-have. |
