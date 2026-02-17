#!/bin/bash
# Quick start script for local testing before deployment

echo "FastFly Local Test Script"
echo "========================="
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.11+"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA GPU not detected. CUDA features will not work."
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r <(cat pyproject.toml | grep -A 100 "dependencies" | grep -E "^\s+\"" | sed 's/[",]//g' | tr -d ' ')

# Check if neuron_annotations.npz exists
if [ ! -f "neuron_annotations.npz" ]; then
    echo "⚠️  neuron_annotations.npz not found"
    echo "   Running with synthetic data..."
fi

# Start the server
echo ""
echo "🚀 Starting FastFly server..."
echo "   Local URL: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

python app.py --host 127.0.0.1 --port 8000
