"""Create a 3-minute teaser video from project artifacts.

Combines title card, data pipeline, denoising GIF frames, results,
exploration demo, and conclusion into a single MP4.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess


def text_frame(text, subtitle="", width=1920, height=1080, bg=(13, 17, 23),
               text_color=(230, 237, 243), accent=(0, 255, 157)):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
        font_med = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font_large = ImageFont.load_default()
        font_med = font_large
        font_small = font_large

    bbox = draw.textbbox((0, 0), text, font=font_large)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, height // 2 - 80), text, fill=accent, font=font_large)

    if subtitle:
        bbox2 = draw.textbbox((0, 0), subtitle, font=font_med)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((width - tw2) // 2, height // 2 + 40), subtitle, fill=text_color, font=font_med)

    return img


def image_frame(image_path, title="", width=1920, height=1080):
    bg = Image.new("RGB", (width, height), (13, 17, 23))

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    title_space = 80 if title else 0
    avail_h = height - title_space - 40
    avail_w = width - 80

    scale = min(avail_w / img_w, avail_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    x = (width - new_w) // 2
    y = title_space + (avail_h - new_h) // 2
    bg.paste(img, (x, y))

    if title:
        draw = ImageDraw.Draw(bg)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) // 2, 20), title, fill=(0, 255, 157), font=font)

    return bg


def gif_to_frames(gif_path, max_frames=30):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(gif.copy().convert("RGB"))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]
    return frames


def gif_frame(gif_path, title="", width=1920, height=1080):
    frames = gif_to_frames(gif_path)
    result = []
    for f in frames:
        bg = Image.new("RGB", (width, height), (13, 17, 23))
        f_w, f_h = f.size
        title_space = 80 if title else 0
        avail_h = height - title_space - 40
        scale = min((width - 80) / f_w, avail_h / f_h)
        new_w, new_h = int(f_w * scale), int(f_h * scale)
        f_resized = f.resize((new_w, new_h), Image.LANCZOS)
        x = (width - new_w) // 2
        y = title_space + (avail_h - new_h) // 2
        bg.paste(f_resized, (x, y))
        if title:
            draw = ImageDraw.Draw(bg)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            except:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) // 2, 20), title, fill=(0, 255, 157), font=font)
        result.append(bg)
    return result


def create_video(results_dir, out_path, fps=2):
    frames = []
    R = results_dir

    # Title (3 seconds = 6 frames at 2fps)
    title = text_frame("Diffusion-Based Map Completion",
                       "for Frontier Exploration")
    for _ in range(6):
        frames.append(title)

    credit = text_frame("Moin Mattar",
                        "COSC 81/281 - Dartmouth College - Spring 2026")
    for _ in range(4):
        frames.append(credit)

    # Problem
    problem = text_frame("The Problem",
                         "Robots explore blindly. What if they could imagine what's behind walls?")
    for _ in range(4):
        frames.append(problem)

    # Pipeline
    frames.append(image_frame(f"{R}/presentation/02_pipeline_diagram.png",
                              "System Pipeline"))
    for _ in range(5):
        frames.append(frames[-1])

    # Data
    data_title = text_frame("Phase 1: Data", "35,126 floor plans - 2.66M training pairs")
    for _ in range(3):
        frames.append(data_title)

    frames.append(image_frame(f"{R}/data_artifacts/05_lidar_raycasting.png",
                              "Lidar Raycasting Simulation"))
    for _ in range(4):
        frames.append(frames[-1])

    frames.append(image_frame(f"{R}/data_artifacts/01_sample_grid_20.png",
                              "Training Samples: Partial Map | Ground Truth | Hidden Regions"))
    for _ in range(4):
        frames.append(frames[-1])

    # Training
    train_title = text_frame("Phase 2: Training", "4.2M param U-Net - 29 epochs - Tesla T4")
    for _ in range(3):
        frames.append(train_title)

    frames.append(image_frame(f"{R}/presentation/01_loss_iou_plot.png",
                              "Training Convergence: Loss 0.033 to 0.0007"))
    for _ in range(5):
        frames.append(frames[-1])

    frames.append(image_frame(f"{R}/showcase/06_training_progression.png",
                              "Predictions Improving Over Training"))
    for _ in range(4):
        frames.append(frames[-1])

    # Denoising GIF
    denoise_title = text_frame("The Denoising Process",
                               "Noise transforms into a predicted map")
    for _ in range(3):
        frames.append(denoise_title)

    gif_frames = gif_frame(f"{R}/gifs/denoise_01.gif",
                           "Denoising: 50 DDIM Steps")
    frames.extend(gif_frames)

    # High coverage denoising
    hc_frames = gif_frame(f"{R}/high_coverage_showcase/hc_denoise_01.gif",
                          "High Coverage Input: Much Sharper Predictions")
    frames.extend(hc_frames)

    # Results
    results_title = text_frame("Results", "82% Accuracy - IoU 0.62 to 0.88")
    for _ in range(4):
        frames.append(results_title)

    frames.append(image_frame(f"{R}/presentation/04_results_poster.png",
                              "Evaluation Summary"))
    for _ in range(6):
        frames.append(frames[-1])

    # Coverage finding
    frames.append(image_frame(f"{R}/coverage_analysis/coverage_comparison_bar.png",
                              "Key Finding: More Coverage = Better Predictions"))
    for _ in range(5):
        frames.append(frames[-1])

    frames.append(image_frame(f"{R}/high_coverage_showcase/05_low_vs_high.png",
                              "1 Scan vs 5 Scans: Two Maps Hit Perfect IoU"))
    for _ in range(5):
        frames.append(frames[-1])

    # Diversity
    frames.append(image_frame(f"{R}/high_coverage_showcase/03_diversity.png",
                              "Sample Diversity: 8 Different Plausible Completions"))
    for _ in range(5):
        frames.append(frames[-1])

    # Uncertainty
    frames.append(image_frame(f"{R}/coverage_artifacts/05_confidence.png",
                              "Uncertainty Decreases as Robot Explores More"))
    for _ in range(5):
        frames.append(frames[-1])

    # Comparison
    comp_title = text_frame("Diffusion vs Baseline", "+24.5% Information Gain")
    for _ in range(4):
        frames.append(comp_title)

    frames.append(image_frame(f"{R}/comparison/comparison_grid.png",
                              "Orange = Baseline Choice, Green = Diffusion Choice"))
    for _ in range(5):
        frames.append(frames[-1])

    # Exploration
    explore_title = text_frame("Full Exploration Demo",
                               "Robot explores a maze using diffusion-guided frontiers")
    for _ in range(3):
        frames.append(explore_title)

    explore_frames = gif_frame(f"{R}/exploration_demo/exploration_diffusion.gif",
                               "Diffusion-Guided Exploration: 91% Coverage")
    frames.extend(explore_frames)

    # Stage maze
    maze_frames = gif_frame(f"{R}/stage_demo/maze_exploration.gif",
                            "Stage Maze: Diffusion-Guided Frontier Selection")
    frames.extend(maze_frames)

    # Exploration story
    frames.append(image_frame(f"{R}/coverage_artifacts/06_exploration_story.png",
                              "The Exploration Story: First Scan to Complete Map"))
    for _ in range(5):
        frames.append(frames[-1])

    # Conclusion
    conclusion = text_frame("Conclusion",
                            "Diffusion models enable smarter robot exploration")
    for _ in range(4):
        frames.append(conclusion)

    results_summary = text_frame("82% Accuracy | IoU 0.88 | +24.5% vs Baseline",
                                 "Model predicts what's behind walls - robots explore smarter")
    for _ in range(5):
        frames.append(results_summary)

    thanks = text_frame("Thank You",
                        "github.com/mo-root/Robots_diffusion_planner")
    for _ in range(6):
        frames.append(thanks)

    # Save frames
    frame_dir = os.path.join(os.path.dirname(out_path), "video_frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(frame_dir, f"frame_{i:04d}.png"))

    # Compile to MP4
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-profile:v", "baseline",
        "-pix_fmt", "yuv420p", "-r", "30",
        "-movflags", "+faststart",
        out_path
    ], capture_output=True, timeout=60)

    duration = len(frames) / fps
    print(f"Video saved: {out_path}")
    print(f"  {len(frames)} frames at {fps} fps = {duration:.0f}s ({duration/60:.1f} min)")


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    out_path = os.path.join(results_dir, "teaser_video.mp4")
    create_video(results_dir, out_path, fps=2)
