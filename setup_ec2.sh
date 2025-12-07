#!/bin/bash

# setup_ec2.sh
# Script to setup the environment and download assets on an EC2 instance.

# 1. System Updates & Dependencies
echo "Subdomain: Updating system..."
sudo yum update -y  # For Amazon Linux
# sudo apt-get update && sudo apt-get upgrade -y # For Ubuntu

# Install Python and pip if needed (Amazon Linux 2023 usually has python3)
if ! command -v python3 &> /dev/null; then
    sudo yum install python3 -y
    # sudo apt-get install python3 python3-pip -y
fi

# 2. Python Dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# 3. Create Directories
echo "Creating data directories..."
mkdir -p data/val/images
mkdir -p data/val/depths
mkdir -p distance_est/ckpt
mkdir -p inside_pred/ckpt

# 4. Download Assets from S3
# Assumes the instance has an IAM role with S3 access attached.
BUCKET="s3://spatial-agent-data-learner-lab"

echo "Downloading Data from $BUCKET..."
# Sync validation images
aws s3 sync $BUCKET/val/images .data/val/images --quiet
aws s3 sync $BUCKET/val/depths .data/val/depths --quiet

# Download Checkpoints
echo "Downloading Checkpoints..."
aws s3 cp $BUCKET/distance_est/ckpt/epoch_5_iter_6831.pth distance_est/ckpt/
aws s3 cp $BUCKET/distance_est/ckpt/3m_epoch6.pth distance_est/ckpt/
aws s3 cp $BUCKET/inside_pred/ckpt/epoch_4.pth inside_pred/ckpt/

echo "Setup Complete!"
echo "Run the app with: streamlit run app.py"
