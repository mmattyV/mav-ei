# 🚀 Lunar Lander + SDP Quick Start

## One-Command Setup & Train

```bash
# Navigate to the repo
cd /Users/matthewvu/Documents/mav-ei/experiments/matt_implementation/streaming_diffusion_policy

# Run automated setup (installs deps + collects data)
./setup_lunar_lander.sh

# Start training
python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander
```

That's it! Training will run for ~2-3 hours.

---

## Manual Step-by-Step

### 1. Install Dependencies (5 min)
```bash
pip install gymnasium[box2d] stable-baselines3 huggingface-sb3 zarr wandb hydra-core omegaconf
```

### 2. Verify Setup (1 min)
```bash
python verify_lunar_lander_setup.py
```

### 3. Collect Data (15 min)
```bash
python scripts/collect_lunar_lander_demos.py --n_episodes 200
```

### 4. Train (2-3 hours)
```bash
python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander
```

---

## What to Expect

### Data Collection
- ⏱️ **Time**: 15-20 minutes
- 📊 **Output**: 200 successful episodes (~40k steps)
- 📁 **Saved to**: `data/lunar_lander/demonstrations.zarr`

### Training
- ⏱️ **Time**: 2-3 hours (GPU) or 20-30 hours (CPU)
- 🎯 **Target**: Test mean score 200+ (solved)
- 📁 **Outputs**: 
  - Checkpoints: `data/outputs/.../checkpoints/`
  - Videos: `data/outputs/.../media/`
  - Logs: W&B dashboard

### Performance
- **Epoch 100**: ~100-150 mean score
- **Epoch 200**: ~150-200 mean score
- **Epoch 400**: ~200-250 mean score (solved!)

---

## Common Commands

### Check training progress
```bash
# View logs
tail -f data/outputs/*/logs.json.txt

# Or check Weights & Biases dashboard
```

### Resume training
```bash
# Training auto-resumes from latest checkpoint if interrupted
python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander
```

### Evaluate trained model
```bash
python eval.py --checkpoint data/outputs/.../checkpoints/latest.ckpt
```

### Collect more data
```bash
python scripts/collect_lunar_lander_demos.py --n_episodes 100
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named gymnasium` | `pip install gymnasium[box2d]` |
| `No module named stable_baselines3` | `pip install stable-baselines3 huggingface-sb3` |
| NumPy version error | `pip install 'numpy<2'` |
| CUDA out of memory | Reduce `batch_size` to 128 in config |
| Training too slow | Check GPU with `nvidia-smi`, or use smaller model |

---

## Files Created

```
✅ scripts/collect_lunar_lander_demos.py       # Data collection
✅ diffusion_policy/dataset/lunar_lander_dataset.py  # Dataset
✅ diffusion_policy/env_runner/lunar_lander_runner.py  # Env runner
✅ diffusion_policy/config/task/lunar_lander_lowdim.yaml  # Task config
✅ diffusion_policy/config/train_tedi_unet_lunar_lander.yaml  # Training config
✅ setup_lunar_lander.sh                       # Automated setup
✅ verify_lunar_lander_setup.py                # Verification
✅ LUNAR_LANDER_README.md                      # Full documentation
✅ SETUP_SUMMARY.md                            # Implementation summary
✅ QUICKSTART.md                               # This file
```

---

## Next: MAV-EI Integration

Once trained, modify `tedi_unet_lowdim_policy.py` to generate K proposals:

```python
def predict_action_samples(self, obs_dict, K=10):
    """Sample K action trajectories"""
    samples = []
    for _ in range(K):
        result = self.predict_action(obs_dict)
        samples.append(result['action'])
        self.reset_buffer()
    return torch.stack(samples, dim=0)
```

Then implement your Best-of-N baseline with verifiers!

---

## Questions?

- 📖 **Full docs**: See `LUNAR_LANDER_README.md`
- 📊 **Implementation details**: See `SETUP_SUMMARY.md`
- 🐛 **Issues**: Check the troubleshooting sections
- 💬 **Help**: Open an issue on GitHub

**Ready to go? Run `./setup_lunar_lander.sh` to get started! 🚀**



