# Conda Environment Setup for Lunar Lander + SDP

## Quick Start

### One-Command Setup
```bash
cd /Users/matthewvu/Documents/mav-ei/experiments/matt_implementation/streaming_diffusion_policy
./setup_conda_env.sh
```

This script will:
1. ✅ Create a conda environment named `sdp_lunar_lander`
2. ✅ Install all required dependencies
3. ✅ Install Box2D for Lunar Lander physics
4. ✅ Verify all installations
5. ✅ Test the Lunar Lander environment

**Time:** ~5-10 minutes depending on your internet connection.

---

## Manual Setup

If the automated script doesn't work, follow these steps:

### Step 1: Create Environment
```bash
conda env create -f conda_environment_lunar_lander.yaml
```

### Step 2: Activate Environment
```bash
conda activate sdp_lunar_lander
```

### Step 3: Install Box2D
```bash
# Try pip first
pip install box2d-py==2.3.5

# If that fails, try conda
conda install -c conda-forge box2d-py -y
```

### Step 4: Verify Installation
```bash
python -c "import gymnasium; import stable_baselines3; print('✓ All imports successful!')"
```

### Step 5: Test Lunar Lander
```bash
python -c "import gymnasium as gym; env = gym.make('LunarLanderContinuous-v3'); env.reset(); print('✓ Lunar Lander working!')"
```

---

## Common Commands

### Activate Environment
```bash
conda activate sdp_lunar_lander
```

### Deactivate Environment
```bash
conda deactivate
```

### List Installed Packages
```bash
conda list
```

### Update Package
```bash
conda activate sdp_lunar_lander
pip install --upgrade <package_name>
```

### Remove Environment
```bash
conda env remove -n sdp_lunar_lander
```

### Export Environment (for sharing)
```bash
conda activate sdp_lunar_lander
conda env export > conda_environment_exported.yaml
```

---

## What's Included

### Core Dependencies
- **Python 3.9** - Base Python version
- **PyTorch 1.12.1** - Deep learning framework
- **NumPy 1.23.3** - Numerical computing

### SDP Dependencies
- **Hydra 1.2.0** - Configuration management
- **Zarr 2.12.0** - Data storage format
- **Wandb 0.13.3** - Experiment tracking
- **Diffusers 0.11.1** - Diffusion models

### Lunar Lander Dependencies
- **Gymnasium 0.29.1** - RL environment (newer version of OpenAI Gym)
- **Box2D** - 2D physics engine for Lunar Lander
- **Stable-Baselines3 2.1.0** - Pre-trained RL algorithms
- **Hugging Face SB3 3.0** - Load models from Hugging Face Hub

### Utilities
- **tqdm** - Progress bars
- **matplotlib** - Plotting
- **opencv** - Image processing

---

## Collecting Demonstrations

Once the environment is set up:

```bash
# Activate environment
conda activate sdp_lunar_lander

# Navigate to repo
cd /Users/matthewvu/Documents/mav-ei/experiments/matt_implementation/streaming_diffusion_policy

# Collect 200 successful episodes (~15 minutes)
python scripts/collect_lunar_lander_demos.py --n_episodes 200

# Output will be saved to: data/lunar_lander/demonstrations.zarr
```

---

## Training SDP

```bash
# Activate environment
conda activate sdp_lunar_lander

# Start training
python train.py \
    --config-dir=diffusion_policy/config/ \
    --config-name=train_tedi_unet_lunar_lander
```

---

## Troubleshooting

### Issue: Conda command not found
**Solution:**
```bash
# Add conda to PATH (replace with your conda installation path)
export PATH="$HOME/anaconda3/bin:$PATH"

# Or use the full path
/Users/matthewvu/anaconda3/bin/conda env create -f conda_environment_lunar_lander.yaml
```

### Issue: Box2D installation fails
**Symptoms:** `ImportError: cannot import name 'Box2D' from 'Box2D'`

**Solutions:**

#### Option 1: Install via conda-forge
```bash
conda activate sdp_lunar_lander
conda install -c conda-forge box2d-py -y
```

#### Option 2: Install SWIG first (macOS)
```bash
brew install swig
conda activate sdp_lunar_lander
pip install box2d-py
```

#### Option 3: Build from source
```bash
conda activate sdp_lunar_lander
pip install swig
pip install box2d-py --no-binary box2d-py
```

### Issue: "No module named 'gymnasium'"
**Solution:**
```bash
conda activate sdp_lunar_lander
pip install gymnasium[box2d]
```

### Issue: PyTorch/CUDA version mismatch
**Solution:**

For CPU-only (macOS):
```bash
# Already configured in conda_environment_lunar_lander.yaml
# PyTorch 1.12.1 without CUDA
```

For GPU (if you have NVIDIA GPU on Linux):
```bash
conda install pytorch=1.12.1 cudatoolkit=11.6 -c pytorch
```

### Issue: "DLL load failed" or "Symbol not found" (macOS)
**Solution:**
```bash
conda activate sdp_lunar_lander
# Reinstall numpy with lower version
conda install numpy=1.23.3 --force-reinstall
```

### Issue: Environment creation is slow
**Tip:** Use mamba (faster conda alternative)
```bash
# Install mamba
conda install mamba -c conda-forge

# Use mamba instead of conda
mamba env create -f conda_environment_lunar_lander.yaml
```

### Issue: Conflicts during installation
**Solution:** Create minimal environment first, then add packages
```bash
# Create minimal environment
conda create -n sdp_lunar_lander python=3.9 -y
conda activate sdp_lunar_lander

# Install PyTorch first
conda install pytorch=1.12.1 -c pytorch -y

# Install other dependencies
pip install gymnasium[box2d] stable-baselines3 huggingface-sb3 zarr wandb hydra-core
```

---

## Verifying Installation

### Quick Test
```bash
conda activate sdp_lunar_lander
python verify_lunar_lander_setup.py
```

Expected output:
```
✓ All required checks passed!
```

### Manual Verification

Test each component individually:

```bash
conda activate sdp_lunar_lander

# Test Python version
python --version  # Should be 3.9.x

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test Gymnasium
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"

# Test Box2D
python -c "import Box2D; print('Box2D: OK')"

# Test Stable Baselines 3
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"

# Test Hugging Face SB3
python -c "import huggingface_sb3; print('HF-SB3: OK')"

# Test Zarr
python -c "import zarr; print(f'Zarr: {zarr.__version__}')"

# Test Hydra
python -c "import hydra; print(f'Hydra: {hydra.__version__}')"

# Test Wandb
python -c "import wandb; print(f'Wandb: {wandb.__version__}')"
```

### Test Lunar Lander Environment
```bash
conda activate sdp_lunar_lander

python << 'EOF'
import gymnasium as gym
import numpy as np

print("Creating Lunar Lander environment...")
env = gym.make('LunarLanderContinuous-v3')

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

print("\nRunning test episode...")
obs, info = env.reset()
print(f"Initial obs: {obs}")

total_reward = 0
for step in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        break

print(f"\nTest completed!")
print(f"Steps: {step + 1}")
print(f"Total reward: {total_reward:.2f}")
print("✓ Lunar Lander working correctly!")

env.close()
EOF
```

---

## Environment Info

### Environment Location
```bash
conda env list | grep sdp_lunar_lander
```

### Disk Space
The environment will take approximately **3-4 GB** of disk space.

### Python Packages Count
The environment includes **~100 packages** (including dependencies).

---

## Next Steps

After successful setup:

1. **Collect Data** (~15 minutes)
   ```bash
   conda activate sdp_lunar_lander
   python scripts/collect_lunar_lander_demos.py --n_episodes 200
   ```

2. **Verify Data** 
   ```bash
   python -c "import zarr; z = zarr.open('data/lunar_lander/demonstrations.zarr', 'r'); print(f'Episodes: {len(z.meta.episode_ends)}'); print(f'Total steps: {len(z.obs)}')"
   ```

3. **Start Training** (~2-3 hours on GPU)
   ```bash
   conda activate sdp_lunar_lander
   python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander
   ```

---

## Useful Links

- **Conda Cheat Sheet**: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Box2D**: https://box2d.org/documentation/

---

## FAQ

**Q: Can I use this environment for other RL tasks?**
A: Yes! It includes Gymnasium and Stable-Baselines3, so you can train on many RL environments.

**Q: Do I need a GPU?**
A: For data collection, no. For training SDP, GPU is highly recommended (50x faster).

**Q: Can I install additional packages?**
A: Yes, activate the environment and use `pip install <package>` or `conda install <package>`.

**Q: How do I update the environment?**
A: Edit `conda_environment_lunar_lander.yaml` and run `conda env update -f conda_environment_lunar_lander.yaml`.

**Q: Can I use Python 3.10 or 3.11?**
A: Some packages may not be compatible. Stick with Python 3.9 for best compatibility.

**Q: Why not use venv instead of conda?**
A: Conda handles non-Python dependencies (like Box2D's C++ libraries) better than venv.

---

## Files Created

```
✅ conda_environment_lunar_lander.yaml   # Environment specification
✅ setup_conda_env.sh                    # Automated setup script
✅ CONDA_SETUP.md                        # This documentation
```

---

**Ready to start? Run `./setup_conda_env.sh` now! 🚀**


