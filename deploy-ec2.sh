#!/bin/bash
# Quick deployment script for EC2

set -e

echo "Building and deploying FastFly to EC2..."

# Build Docker image
echo "Building Docker image..."
docker compose build

# Stop existing container
echo "Stopping existing containers..."
docker compose down

# Start new container
echo "Starting new container..."
docker compose up -d

# Show logs
echo ""
echo "Container started! Showing logs (Ctrl+C to exit):"
docker compose logs -f
