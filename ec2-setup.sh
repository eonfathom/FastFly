#!/bin/bash
# EC2 GPU Instance Setup Script for FlyWire Simulator
# Run this on a fresh EC2 GPU instance (g4dn.xlarge or similar)

set -e

echo "================================"
echo "FlyWire Simulator - EC2 Setup"
echo "================================"

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Install Docker Compose
echo "Installing Docker Compose..."
sudo apt-get install -y docker-compose-plugin

# Verify GPU access
echo "Verifying GPU access..."
nvidia-smi

# Clone repository (if not already present)
if [ ! -d "FastFly" ]; then
    echo "Clone your FastFly repository here or upload files"
    # git clone <your-repo-url> FastFly
    # cd FastFly
fi

# Generate self-signed SSL certificate (optional, for HTTPS)
# Uncomment if you want to use HTTPS
# sudo apt-get install -y certbot
# sudo certbot certonly --standalone -d your-domain.com

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Upload your FastFly code to this instance"
echo "2. Set ALLOWED_ORIGINS in .env file"
echo "3. Run: docker compose up -d"
echo "4. Configure security group to allow port 8000"
echo ""
echo "To test GPU in Docker:"
echo "  docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi"
echo ""
