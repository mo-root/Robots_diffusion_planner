# Diffusion-Based Map Completion for Frontier Exploration

**COSC 81/281 Final Project** -- Dartmouth College, Spring 2026

**Author:** Moin Mattar

## Overview

A conditional denoising diffusion model (DDPM) that predicts complete occupancy grid maps from partial lidar observations. The predicted maps are used to score frontier cells for smarter autonomous exploration.

```
Partial Map (lidar)     Diffusion Model        Scored Frontiers
 ████████████            ████████████            ████████████
 █·····██···█            █·····██···█            █·····██···█
 █·····██···█    --->    █·····██···█    --->    █·····██···█
 █··R··??????            █··R··██···█            █··R··██···█
 █·····??????            █·····█    █            █·····█  B █  <-- go here
 ████████████            ████████████            ████████████
    Known only            Predicted full          Best frontier
```

The model generates **multiple plausible completions** (K=8 samples), and frontiers are scored by expected information gain + uncertainty bonus. This replaces the standard "biggest unknown patch" heuristic with a learned structural prior.

## Architecture

| Component | Description |
|---|---|
| **Model** | Conditional U-Net (~4.3M params) with sinusoidal time embeddings |
| **Training** | DDPM (Ho et al. 2020), 1000 diffusion steps, MSE noise prediction loss |
| **Inference** | DDIM sampling (50 steps) for fast generation |
| **Data** | HouseExpo dataset (35k floor plans) with synthetic lidar simulation |
| **Integration** | ROS 2 Humble node, connects to PA4 mapper + PA3 planner |

## Project Structure

```
.
├── src/                    # Core source code
│   ├── data_generator.py   # HouseExpo -> (partial, full) training pairs
│   ├── dataset.py          # PyTorch Dataset loader
│   ├── unet.py             # Conditional U-Net architecture
│   ├── diffusion.py        # DDPM forward/reverse process
│   └── train.py            # Training loop with logging
├── scripts/                # Utility scripts (AWS setup, data download)
├── configs/                # Training configurations
├── tests/                  # Unit tests
├── data/                   # Dataset (not tracked in git)
├── results/                # Training outputs
│   ├── samples/            # Generated sample predictions per epoch
│   ├── loss_curves/        # Loss plots
│   └── checkpoints/        # Model weights
├── docs/                   # Proposal, report, presentation
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and prepare data

```bash
# Clone HouseExpo dataset
git clone https://github.com/TeaganLi/HouseExpo.git data/HouseExpo
cd data/HouseExpo/HouseExpo && tar -xzf json.tar.gz && cd ../../..

# Generate training pairs (~2M samples with augmentation)
python src/data_generator.py \
    --json_dir data/HouseExpo/HouseExpo/json \
    --out_dir data/train \
    --val_dir data/val \
    --samples_per_map 10 \
    --num_workers 8
```

### 3. Train

```bash
# Local (Apple Silicon)
python src/train.py --train_dir data/train --val_dir data/val --device mps --batch_size 8 --epochs 100

# AWS GPU
python src/train.py --train_dir data/train --val_dir data/val --device cuda --batch_size 64 --epochs 100
```

### 4. Evaluate

```bash
python src/evaluate.py --checkpoint results/checkpoints/model_final.pt --test_dir data/val
```

## Method

### Data Generation Pipeline

1. Load floor plan polygons from HouseExpo (35,126 indoor layouts)
2. Rasterize to 256x256 binary occupancy grids
3. Place robot at random free-space position
4. Simulate 360-degree lidar raycasting (variable range 40-100px)
5. Build partial map (visible cells = known, rest = unknown)
6. Augment with 90/180/270 rotations + horizontal flip (8x data)

### Diffusion Model

- **Forward process:** Gradually add Gaussian noise to ground-truth map over T=1000 steps
- **Reverse process:** U-Net learns to predict and remove noise, conditioned on partial map + known mask
- **Sampling:** DDIM with 50 steps for ~20x faster inference than full DDPM

### Frontier Scoring

For each frontier cell on the known/unknown boundary:
1. Sample K=8 map completions from the diffusion model
2. For each completion, count newly revealed free cells (information gain)
3. Score = E[info_gain] + lambda * Std[info_gain] - beta * distance

The variance bonus encourages exploring uncertain areas where the model disagrees.

## Baselines

1. **Heuristic frontier** -- score = nearby unknown cells - distance (no learning)
2. **Deterministic U-Net** -- same architecture, MSE regression, single prediction
3. **Diffusion (no variance)** -- expected gain only, no uncertainty bonus
4. **Diffusion (full)** -- expected gain + variance bonus (ours)

## Timeline

| Date | Milestone |
|---|---|
| May 23 | Data pipeline complete, training started |
| May 27 | Progress check-in: loss curves + sample predictions |
| June 1 | Final presentation (6 min) |
| June 6 | Final report + code + demo video |

## References

- Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al. "Denoising Diffusion Implicit Models" (ICLR 2021)
- Li et al. "HouseExpo: A Large-scale 2D Indoor Layout Dataset" (IROS 2020)
- Shrestha et al. "Learned Map Prediction for Enhanced Mobile Robot Exploration" (ICRA 2019)
- Lin et al. "Online Diffusion-Based 3D Occupancy Prediction at the Frontier" (arXiv 2024)

## License

MIT
