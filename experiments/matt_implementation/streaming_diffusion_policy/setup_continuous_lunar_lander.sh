#!/bin/bash
# Complete setup script for continuous LunarLander with SDP
# This script:
#   1. Trains a continuous PPO model (~20-30 min)
#   2. Collects expert demonstrations using the trained model
#   3. Prepares data for SDP training

set -e  # Exit on error

echo "========================================================================"
echo "Continuous LunarLander Setup for Streaming Diffusion Policy"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Train a continuous PPO model for LunarLanderContinuous-v3 (~20-30 min)"
echo "  2. Collect 200 expert demonstrations"
echo "  3. Prepare data for SDP training"
echo ""
echo "Prerequisites:"
echo "  - Conda environment 'sdp_lunar_lander' must be activated"
echo "  - stable-baselines3 must be installed"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Check if conda env is activated
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ Error: No conda environment activated"
    echo "Please run: conda activate sdp_lunar_lander"
    exit 1
fi

if [[ "$CONDA_DEFAULT_ENV" != "sdp_lunar_lander" ]]; then
    echo "⚠️  Warning: Current environment is '$CONDA_DEFAULT_ENV', expected 'sdp_lunar_lander'"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if stable-baselines3 is installed
if ! python -c "import stable_baselines3" 2>/dev/null; then
    echo "❌ Error: stable-baselines3 not installed"
    echo "Installing stable-baselines3..."
    pip install stable-baselines3 huggingface-sb3
fi

echo ""
echo "========================================================================"
echo "Step 1/3: Training Continuous PPO Model"
echo "========================================================================"
echo ""
echo "Training will take approximately 20-30 minutes on CPU (5-10 min on GPU)"
echo "You can monitor progress with TensorBoard:"
echo "  tensorboard --logdir models/lunar_lander_continuous/tensorboard"
echo ""

# Train the model
python scripts/train_continuous_lunar_lander.py \
    --timesteps 500000 \
    --output models/lunar_lander_continuous \
    --eval_freq 10000

# Check if training succeeded
if [ ! -f "models/lunar_lander_continuous/final_model.zip" ]; then
    echo "❌ Error: Training failed - model file not found"
    exit 1
fi

echo ""
echo "✓ Training complete!"
echo ""
echo "========================================================================"
echo "Step 2/3: Collecting Expert Demonstrations"
echo "========================================================================"
echo ""

# Collect demonstrations using the trained model
python scripts/collect_lunar_lander_demos.py \
    --model models/lunar_lander_continuous/final_model.zip \
    --n_episodes 200 \
    --min_reward 200 \
    --output data/lunar_lander/demonstrations.zarr

# Check if data collection succeeded
if [ ! -d "data/lunar_lander/demonstrations.zarr" ]; then
    echo "❌ Error: Data collection failed - zarr file not found"
    exit 1
fi

echo ""
echo "✓ Data collection complete!"
echo ""
echo "========================================================================"
echo "Step 3/3: Verifying Setup"
echo "========================================================================"
echo ""

# Run verification
if [ -f "verify_lunar_lander_setup.py" ]; then
    python verify_lunar_lander_setup.py
else
    echo "⚠️  Warning: Verification script not found, skipping..."
fi

echo ""
echo "========================================================================"
echo "Setup Complete! 🎉"
echo "========================================================================"
echo ""
echo "You can now train SDP with:"
echo "  python train.py --config-name train_tedi_unet_lunar_lander"
echo ""
echo "Files created:"
echo "  - models/lunar_lander_continuous/final_model.zip (trained PPO model)"
echo "  - models/lunar_lander_continuous/best/best_model.zip (best checkpoint)"
echo "  - data/lunar_lander/demonstrations.zarr (expert demonstrations)"
echo ""
echo "Monitor training with:"
echo "  tensorboard --logdir data/outputs"
echo ""

