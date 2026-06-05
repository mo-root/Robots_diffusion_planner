#!/usr/bin/env bash
# Launch the full diffusion-guided exploration system in Stage.
#
# Components:
#   1. Stage simulator (maze world)
#   2. PA4 occupancy grid mapper
#   3. Diffusion frontier scorer (or baseline with --baseline flag)
#   4. Exploration manager (drives robot to scored frontiers)
#
# Usage:
#   bash launch_exploration.sh                    # diffusion-guided
#   bash launch_exploration.sh --baseline         # heuristic baseline
#   bash launch_exploration.sh --checkpoint /path/to/model.pt

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CHECKPOINT="${PROJECT_DIR}/results/checkpoints/model_epoch0020.pt"
MODE="diffusion"
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline) MODE="baseline"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source /opt/ros/humble/setup.bash
[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash

PIDS=()
cleanup() {
    echo "[cleanup] Stopping all processes..."
    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    sleep 1
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

echo "============================================"
echo "  Diffusion-Guided Exploration System"
echo "  Mode: $MODE"
echo "============================================"

echo "[1/4] Launching Stage simulator..."
ros2 launch stage_ros2 stage.launch.py \
    world:=/root/ros2_ws/src/pa3/maze \
    enforce_prefixes:=false one_tf_tree:=true \
    > /tmp/stage.log 2>&1 &
PIDS+=($!)
sleep 5

echo "[2/4] Launching PA4 occupancy grid mapper..."
python3 /root/ros2_ws/src/pa4/pa4_grid_mapper.py --ros-args \
    -p resolution:=0.05 \
    -p update_mode:=log_odds \
    -p base_frame:=rosbot/base_link \
    -p odom_frame:=rosbot/odom \
    -p scan_topic:=/base_scan \
    -p publish_rate:=2.0 \
    > /tmp/mapper.log 2>&1 &
PIDS+=($!)
sleep 3

if [ "$MODE" = "diffusion" ]; then
    echo "[3/4] Launching diffusion frontier scorer..."
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Device: $DEVICE"
    python3 "$SCRIPT_DIR/diffusion_frontier_node.py" --ros-args \
        -p checkpoint_path:="$CHECKPOINT" \
        -p device:="$DEVICE" \
        -p K:=8 \
        -p ddim_steps:=50 \
        -p score_interval:=5.0 \
        > /tmp/frontier_scorer.log 2>&1 &
else
    echo "[3/4] Launching baseline frontier scorer..."
    python3 "$SCRIPT_DIR/baseline_frontier_node.py" --ros-args \
        -p score_interval:=5.0 \
        > /tmp/frontier_scorer.log 2>&1 &
fi
PIDS+=($!)
sleep 2

echo "[4/4] Launching exploration manager..."
python3 "$SCRIPT_DIR/exploration_manager.py" --ros-args \
    -p max_linear_speed:=0.2 \
    -p max_angular_speed:=0.5 \
    -p goal_tolerance:=0.5 \
    -p coverage_threshold:=0.90 \
    -p odom_frame:=rosbot/odom \
    -p occ_threshold:=50 \
    -p inflate_radius:=3 \
    > /tmp/explorer.log 2>&1 &
PIDS+=($!)

echo ""
echo "============================================"
echo "  All systems running!"
echo "  Mode: $MODE"
echo "  Logs: /tmp/{stage,mapper,frontier_scorer,explorer}.log"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop."

wait
