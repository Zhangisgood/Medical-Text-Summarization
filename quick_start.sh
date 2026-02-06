#!/bin/bash

# Quick Start Script for Medical Text Summarization Project
# This script sets up the environment and runs the complete pipeline

set -e  # Exit on error

echo "=========================================="
echo "Medical Text Summarization - Quick Start"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Create virtual environment
echo -e "\n${BLUE}[1/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Step 2: Activate virtual environment and install dependencies
echo -e "\n${BLUE}[2/6] Installing dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Prepare dataset
echo -e "\n${BLUE}[3/6] Preparing dataset...${NC}"
if [ ! -d "data/processed/dataset_dict" ]; then
    python src/data_preparation.py --max_samples 1000
    echo -e "${GREEN}✓ Dataset prepared${NC}"
else
    echo -e "${YELLOW}Dataset already exists, skipping preparation${NC}"
fi

# Step 4: Train model (with config1 - balanced settings)
echo -e "\n${BLUE}[4/6] Training model (this may take 10-30 minutes)...${NC}"
if [ ! -d "models/finetuned/final_model" ]; then
    python src/train.py --config config1
    echo -e "${GREEN}✓ Model trained${NC}"
else
    echo -e "${YELLOW}Model already exists, skipping training${NC}"
    read -p "Do you want to retrain? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python src/train.py --config config1
        echo -e "${GREEN}✓ Model retrained${NC}"
    fi
fi

# Step 5: Evaluate model
echo -e "\n${BLUE}[5/6] Evaluating model...${NC}"
python src/evaluate.py --model_path models/finetuned/final_model
echo -e "${GREEN}✓ Evaluation complete${NC}"

# Step 6: Run error analysis
echo -e "\n${BLUE}[6/6] Running error analysis...${NC}"
python src/error_analysis.py --model_path models/finetuned/final_model
echo -e "${GREEN}✓ Error analysis complete${NC}"

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "=========================================="
echo -e "\n${BLUE}Next Steps:${NC}"
echo -e "  1. View results in results/ directory"
echo -e "  2. Run inference: python src/inference.py --interactive"
echo -e "  3. Launch web app: python app.py"
echo -e "  4. View training logs: tensorboard --logdir logs/"
echo -e "\n${BLUE}To launch the web interface:${NC}"
echo -e "  python app.py"
echo -e "\nThen open http://localhost:7860 in your browser"
