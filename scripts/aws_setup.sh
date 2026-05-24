#!/usr/bin/env bash
# Launch an AWS EC2 GPU instance for training.
# Prerequisites: aws sso login --profile codex
#
# Usage: bash scripts/aws_setup.sh
#
# This script:
# 1. Launches a g4dn.xlarge (1x T4 GPU, 4 vCPU, 16 GB RAM, $0.526/hr)
# 2. Installs PyTorch + dependencies
# 3. Clones the repo
# 4. Generates training data on the instance (fast, ~10 min with 4 workers)
# 5. Starts training

set -euo pipefail

PROFILE="${AWS_PROFILE:-codex}"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2 (us-east-1), will be overridden by Deep Learning AMI
KEY_NAME="${KEY_NAME:-}"

echo "============================================"
echo "AWS GPU Training Setup"
echo "============================================"
echo "Profile: $PROFILE"
echo "Region:  $REGION"
echo "Instance: $INSTANCE_TYPE"
echo ""

# Find the latest Deep Learning AMI (PyTorch)
echo "[1/4] Finding latest PyTorch Deep Learning AMI..."
DL_AMI=$(aws ec2 describe-images \
    --profile "$PROFILE" \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=*Deep Learning AMI GPU PyTorch*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text 2>/dev/null || echo "")

if [ -z "$DL_AMI" ] || [ "$DL_AMI" = "None" ]; then
    echo "  Could not find PyTorch DL AMI, using base Amazon Linux 2"
    DL_AMI="$AMI_ID"
else
    echo "  Found: $DL_AMI"
fi

# User data script to run on boot
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e

# Install git-lfs
yum install -y git-lfs || apt-get install -y git-lfs || true

# Clone the project
cd /home/ec2-user || cd /home/ubuntu
git clone https://github.com/mo-root/Robots_diffusion_planner.git
cd Robots_diffusion_planner

# Install deps (DL AMI already has PyTorch)
pip install -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt

# Download HouseExpo
git clone https://github.com/TeaganLi/HouseExpo.git data/HouseExpo
cd data/HouseExpo/HouseExpo && tar -xzf json.tar.gz && cd ../../..

# Generate training data
python src/data_generator.py \
    --json_dir data/HouseExpo/HouseExpo/json \
    --out_dir data/train \
    --val_dir data/val \
    --samples_per_map 10 \
    --num_workers 4

# Start training
nohup python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --device cuda \
    --batch_size 64 \
    --epochs 100 \
    --ckpt_dir results/checkpoints \
    --log_dir results \
    > training.log 2>&1 &

echo "Training started! Check training.log for progress."
USERDATA
)

echo ""
echo "[2/4] Launching instance..."
echo "  AMI: $DL_AMI"
echo "  Type: $INSTANCE_TYPE"

LAUNCH_CMD="aws ec2 run-instances \
    --profile $PROFILE \
    --region $REGION \
    --image-id $DL_AMI \
    --instance-type $INSTANCE_TYPE \
    --count 1 \
    --block-device-mappings '[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=diffusion-map-training}]' \
    --user-data \"$(echo "$USER_DATA" | base64)\""

if [ -n "$KEY_NAME" ]; then
    LAUNCH_CMD="$LAUNCH_CMD --key-name $KEY_NAME"
fi

echo ""
echo "Ready to launch. Run this command:"
echo ""
echo "$LAUNCH_CMD"
echo ""
echo "After launch, connect with:"
echo "  aws ec2 describe-instances --profile $PROFILE --filters 'Name=tag:Name,Values=diffusion-map-training' --query 'Reservations[].Instances[].PublicIpAddress' --output text"
echo "  ssh ec2-user@<IP>"
