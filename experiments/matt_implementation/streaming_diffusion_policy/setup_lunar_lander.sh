#!/bin/bash
# Setup script for Lunar Lander with Streaming Diffusion Policy

set -e  # Exit on error

echo "================================================"
echo "Lunar Lander + SDP Setup Script"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Step 1: Installing dependencies...${NC}"
pip install gymnasium[box2d] stable-baselines3 huggingface-sb3 zarr numcodecs tqdm

echo ""
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

echo -e "${YELLOW}Step 2: Creating data directory...${NC}"
mkdir -p data/lunar_lander

echo -e "${GREEN}✓ Directory created${NC}"
echo ""

echo -e "${YELLOW}Step 3: Collecting demonstration data...${NC}"
echo "This will take about 10-20 minutes to collect 200 successful episodes."
echo ""

python scripts/collect_lunar_lander_demos.py \
    --n_episodes 200 \
    --output data/lunar_lander/demonstrations.zarr \
    --min_reward 200

echo ""
echo -e "${GREEN}✓ Data collection complete${NC}"
echo ""

echo "================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Review the collected data in: data/lunar_lander/demonstrations.zarr"
echo "  2. Start training with:"
echo ""
echo "     python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander"
echo ""
echo "  3. Monitor training on Weights & Biases"
echo ""
echo "================================================"



