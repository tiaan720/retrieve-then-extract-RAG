#!/bin/bash

# Quick Start Script for Weaviate Embedding Pipeline

echo "================================================"
echo "Weaviate Embedding Pipeline - Quick Start"
echo "================================================"

# Check Python version
echo -e "\n✓ Checking Python version..."
python3 --version

# Install dependencies
echo -e "\n✓ Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
echo -e "\n✓ Checking Ollama installation..."
if command -v ollama &> /dev/null
then
    echo "   Ollama is installed"
    echo "   Pulling embedding model..."
    ollama pull nomic-embed-text
else
    echo "   ⚠️  Ollama is not installed"
    echo "   Please install Ollama from: https://ollama.ai/"
fi

# Check if Docker is installed
echo -e "\n✓ Checking Docker installation..."
if command -v docker &> /dev/null
then
    echo "   Docker is installed"
    echo "   Starting Weaviate..."
    docker-compose up -d
    echo "   Waiting for Weaviate to be ready..."
    sleep 5
else
    echo "   ⚠️  Docker is not installed"
    echo "   Please install Docker from: https://www.docker.com/"
fi

# Run demo
echo -e "\n✓ Running pipeline demo..."
python demo.py

echo -e "\n================================================"
echo "Setup Complete!"
echo "================================================"
echo -e "\nTo run the full pipeline:"
echo "  python main.py"
echo -e "\nTo run tests:"
echo "  python test_components.py"
echo -e "\nTo view example usage:"
echo "  python example_usage.py"
echo "================================================"
