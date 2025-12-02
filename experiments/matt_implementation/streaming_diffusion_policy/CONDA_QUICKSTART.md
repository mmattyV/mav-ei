# 🚀 Conda Environment - Quick Start

## One Command to Rule Them All

```bash
cd /Users/matthewvu/Documents/mav-ei/experiments/matt_implementation/streaming_diffusion_policy
./setup_conda_env.sh
```

**That's it!** The script will:
- ✅ Create conda environment `sdp_lunar_lander`
- ✅ Install all dependencies (PyTorch, Gymnasium, SB3, etc.)
- ✅ Install Box2D for Lunar Lander physics
- ✅ Verify everything works
- ✅ Test Lunar Lander environment

**Time:** ~5-10 minutes

---

## After Setup Complete

### 1. Activate Environment
```bash
conda activate sdp_lunar_lander
```

### 2. Collect Expert Demonstrations
```bash
python scripts/collect_lunar_lander_demos.py --n_episodes 200
```
⏱️ Takes ~15 minutes

### 3. Train SDP
```bash
python train.py \
    --config-dir=diffusion_policy/config/ \
    --config-name=train_tedi_unet_lunar_lander
```
⏱️ Takes ~2-3 hours on GPU

---

## Quick Commands

```bash
# Activate environment
conda activate sdp_lunar_lander

# Deactivate
conda deactivate

# Check what's installed
conda list

# Remove environment (if needed)
conda env remove -n sdp_lunar_lander
```

---

## Files Created

✅ **`conda_environment_lunar_lander.yaml`** - Environment specification
✅ **`setup_conda_env.sh`** - Automated setup script  
✅ **`CONDA_SETUP.md`** - Full documentation with troubleshooting
✅ **`CONDA_QUICKSTART.md`** - This file

---

## What Gets Installed

### Core
- Python 3.9
- PyTorch 1.12.1
- NumPy 1.23.3

### RL & Environments
- Gymnasium 0.29.1 (with Box2D)
- Stable-Baselines3 2.1.0
- Hugging Face SB3 3.0

### SDP Dependencies
- Hydra 1.2.0
- Zarr 2.12.0
- Wandb 0.13.3
- Diffusers 0.11.1

---

## Troubleshooting

### Script fails?
See detailed troubleshooting in `CONDA_SETUP.md`

### Box2D won't install?
```bash
# Try this manually:
conda activate sdp_lunar_lander
conda install -c conda-forge box2d-py -y
```

### Need help?
1. Check `CONDA_SETUP.md` for detailed docs
2. Run `python verify_lunar_lander_setup.py` to diagnose issues

---

## Ready?

**Run the setup script now:**
```bash
./setup_conda_env.sh
```

Then follow the on-screen instructions! 🎉



