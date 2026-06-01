# What to say, section by section

Open this on your phone. One block per page section. Read top to bottom.

Total speaking time: about 5 minutes.

## Open (10 seconds)

Hi everyone. I am Moin. My project is Diffusion-Guided Frontier Exploration. I trained a diffusion model on real building floor plans and used it as a structural prior to help a robot decide where to explore next.

## Section 0  ·  Inspiration  (40 seconds)

Think about walking into a friend's house you have never been to. You see a leash by the door, a chewed toy in the corner, and you already predict there is probably dog food in the kitchen and a yard behind the back door. You do not just see what is there, you imagine what is behind the rooms you have not entered.

That is the core idea behind world models. A learned prior over how the world is structured, so an agent can predict the unseen.

The biggest names in AI are building this for production robotics right now. NVIDIA Cosmos was trained on 20 million hours of robot and driving video. DeepMind Genie 2 generates interactive 3D worlds from a single image. Decart shows that real time generative world models on commodity GPUs are now possible.

My project is the simplest possible instance of this paradigm for a class robotics problem. A building world model that helps a robot pick the next frontier.

The hypothesis is conditional. The prior helps only when the training distribution matches deployment, the step budget is tight, and inference is fast enough to drive.

## Section 2  ·  System  (35 seconds)

Four stages.

One, the data. HouseExpo, 35,126 residential floor plans. For each one, a partial view from a simulated lidar plus the hidden remainder. After flips and rotations, 2.66 million training pairs.

Two, the model. A conditional U-Net with 4.16 million parameters. Standard DDPM, MSE on the noise.

Three, the scoring. At each frontier decision I sample K equals 8 plausible completions. The frontier score is expected information gain plus a variance bonus minus distance.

Four, the integration. A ROS 2 node publishes to slash best frontier. The exploration manager from PA3 drives the robot.

## Section 3  ·  Math  (35 seconds)

Standard DDPM forward process. At each step, take a clean floor plan, sample a timestep, add Gaussian noise with the cumulative schedule alpha bar.

The training objective is MSE between the predicted noise and the true noise. The U-Net is conditioned on the partial map and a known mask channel.

The new logic is the frontier scoring rule. For each frontier candidate, the first term is the mean information gain across K imagined buildings. The second term is the variance bonus, which is the upper confidence bound trick from bandits. When K samples disagree, the model is telling us it does not know what is behind that door, so we push the robot toward it. The third term subtracts travel distance. Pick the argmax.

## Section 4  ·  Training  (20 seconds)

29 epochs on a T4 GPU. MSE on noise dropped from 0.033 to 0.0007. Validation IoU averaged 0.66 with the best epoch hitting 0.77.

The sample progression panel shows the model waking up. Epoch 5 is blurry blobs, epoch 20 has clean walls and rooms. I used the epoch 20 checkpoint.

The denoising GIF shows four DDIM samples emerging from pure Gaussian noise into four distinct plausible buildings. Variance between them is exactly the exploration signal.

## Section 5  ·  Demos  (60 seconds)

Five demos.

First, the headline animation. Left is the partial map with scored frontiers. Middle is the model's prediction. Right is ground truth. Far right is coverage climbing. 91 percent in 11 steps.

Second, side by side. Top row is diffusion. Bottom row is the classical heuristic baseline. Same map, same start, same lidar.

Third, four frontier scorers running synchronized on the same map. Top left is nearest. Top right is info gain only. Bottom left is info gain plus distance. Bottom right is ours. Watch the coverage tickers. Diffusion reaches 95 percent by step 4.

Fourth, behind the mind. Top is the world the robot does not see. Middle is its partial view with frontier candidates. Bottom is the imagined building from the K samples with the picked frontier circled.

Fifth, the Stage simulator. The ROS 2 pipeline running in the PA3 maze world. Real ROS 2 nodes on the slash map and slash best frontier topics.

## Section 6  ·  Results  (60 seconds)

Three results.

One, the 4-baseline ablation. Four scorers on 30 held-out HouseExpo maps with identical seeds. The decomposition is clean. Switching from nearest to info gain alone is worth plus 17.5 points at step 4. Adding the distance penalty on top contributes nothing. The diffusion prior adds another plus 3.5 points on top of that. Info gain matters, distance is decorative, the prior adds a small but real early budget head start.

Two, the OOD ablation. I trained a second U-Net from scratch on synthetic warehouse layouts and evaluated it on the same 30 residential maps. The advantage drops from plus 3.5 to negative 0.1 points, statistically zero. That is the kill. The prior is doing real structural work, not averaging.

Three, the realtime upper bound. K equals 4 with DDIM 10 steps measures at about 500 milliseconds on T4, exactly the realtime budget. If we re-sample every three driving cells, the in-domain advantage extrapolates to plus 10 points. Distillation is the path.

## Section 7  ·  Hardships  (20 seconds)

Docker QEMU on Apple Silicon is about 50 times too slow for live diffusion in Stage, so the Stage integration uses pre-computed waypoints. Asymptotic coverage, the baseline catches up by step 20. OOD priors give zero help. Two of 30 maps are hard failures where the model gets stuck. Live hardware deployment is blocked by the same latency wall.

## Section 8  ·  What is next  (15 seconds)

Distillation to a Jetson sized model for sub-second inference. Richer training domains, Gibson and Matterport. K-sample disagreement as an I do not know flag. Live ROSbot deployment in a matched distribution building.

## Close  (5 seconds)

That is the project. Thank you. Happy to take questions.

---

## Total budget check

| Section | Time |
|---|---|
| Open | 0:10 |
| Inspiration | 0:40 |
| System | 0:35 |
| Math | 0:35 |
| Training | 0:20 |
| Demos | 1:00 |
| Results | 1:00 |
| Hardships | 0:20 |
| What is next | 0:15 |
| Close | 0:05 |
| **Total** | **5:00** |

One minute of slack against the 6 minute limit.

## If a GIF does not load

Stay calm. Say "the animation isn't loading, let me describe it" and then read the caption for that demo. Move on.

## One-paragraph fallback if everything breaks

I trained a conditional diffusion U-Net on 35,000 HouseExpo floor plans. At each frontier decision the robot samples K equals 8 plausible completions and scores each frontier by expected information gain plus variance bonus minus distance. On 30 held-out maps this gives a plus 3.5 point early budget advantage at step 4. The OOD ablation, warehouse-trained model on residential maps, drops the advantage to negative 0.1 points, which proves the prior is doing real structural work. If inference were sub-second the realtime advantage extrapolates to plus 10 points. Stage integration runs via pre-computed waypoints because Docker QEMU on Apple Silicon is too slow for live diffusion.
