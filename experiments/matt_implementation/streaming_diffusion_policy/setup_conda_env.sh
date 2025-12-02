#!/bin/bash
# Setup conda environment for Lunar Lander data collection with SDP

set -e  # Exit on error

echo "================================================"
echo "SDP Lunar Lander - Conda Environment Setup"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${BLUE}Conda found: $(conda --version)${NC}"
echo ""

# Environment name
ENV_NAME="sdp_lunar_lander"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

echo -e "${YELLOW}Step 1: Creating conda environment from yaml file...${NC}"
echo "This may take 5-10 minutes..."
echo ""

conda env create -f conda_environment_lunar_lander.yaml

echo ""
echo -e "${GREEN}✓ Conda environment created${NC}"
echo ""

echo -e "${YELLOW}Step 2: Activating environment and installing Box2D...${NC}"
echo ""

# Activate environment and install Box2D
# Note: We need to do this in a subshell because conda activate doesn't work well in scripts
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install Box2D specifically (sometimes needs special handling on macOS)
echo "Installing Box2D for Lunar Lander physics engine..."

# Try to install box2d-py via pip
pip install box2d-py==2.3.5 || {
    echo -e "${YELLOW}Warning: box2d-py installation failed. Trying alternative method...${NC}"
    
    # On macOS, might need to install via conda-forge
    conda install -c conda-forge box2d-py -y || {
        echo -e "${RED}Box2D installation failed. You may need to install it manually.${NC}"
        echo "Try: conda activate ${ENV_NAME} && pip install box2d-py"
    }
}

echo ""
echo -e "${GREEN}✓ Box2D installed${NC}"
echo ""

echo -e "${YELLOW}Step 3: Verifying installation...${NC}"
echo ""

# Test imports
python -c "
import sys
print('Testing imports...')

try:
    import gymnasium
    print('✓ gymnasium')
except ImportError as e:
    print(f'✗ gymnasium: {e}')
    sys.exit(1)

try:
    import gymnasium.envs.box2d
    print('✓ gymnasium.envs.box2d')
except ImportError as e:
    print(f'✗ gymnasium.envs.box2d: {e}')
    sys.exit(1)

try:
    import stable_baselines3
    print('✓ stable_baselines3')
except ImportError as e:
    print(f'✗ stable_baselines3: {e}')
    sys.exit(1)

try:
    import huggingface_sb3
    print('✓ huggingface_sb3')
except ImportError as e:
    print(f'✗ huggingface_sb3: {e}')
    sys.exit(1)

try:
    import zarr
    print('✓ zarr')
except ImportError as e:
    print(f'✗ zarr: {e}')
    sys.exit(1)

try:
    import torch
    print('✓ torch')
except ImportError as e:
    print(f'✗ torch: {e}')
    sys.exit(1)

print('')
print('All imports successful!')
" || {
    echo -e "${RED}Import verification failed!${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

echo -e "${YELLOW}Step 4: Testing Lunar Lander environment...${NC}"
echo ""

python -c "
import gymnasium as gym
print('Creating Lunar Lander environment...')
env = gym.make('LunarLanderContinuous-v3')
print(f'✓ Environment created successfully')
print(f'  Observation space: {env.observation_space}')
print(f'  Action space: {env.action_space}')

obs, info = env.reset()
print(f'✓ Environment reset successful')
print(f'  Initial observation shape: {obs.shape}')

action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
print(f'✓ Environment step successful')
print(f'  Step reward: {reward:.2f}')

env.close()
print('✓ Lunar Lander environment working correctly!')
" || {
    echo -e "${RED}Lunar Lander test failed!${NC}"
    echo "This might be a Box2D installation issue."
    exit 1
}

echo ""
echo -e "${GREEN}✓ Lunar Lander environment tested successfully${NC}"
echo ""

echo "================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Environment name: ${ENV_NAME}"
echo ""
echo "To activate the environment:"
echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
echo ""
echo "To collect demonstrations:"
echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
echo -e "  ${BLUE}python scripts/collect_lunar_lander_demos.py --n_episodes 200${NC}"
echo ""
echo "To train SDP:"
echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
echo -e "  ${BLUE}python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander${NC}"
echo ""
echo "================================================"


