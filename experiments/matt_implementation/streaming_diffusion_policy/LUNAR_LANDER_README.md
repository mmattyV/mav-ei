# Lunar Lander with Streaming Diffusion Policy

This guide shows you how to train and evaluate SDP on the Lunar Lander environment.

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup_lunar_lander.sh

# Run automated setup (installs deps + collects data)
./setup_lunar_lander.sh

# Start training
python train.py --config-dir=diffusion_policy/config/ --config-name=train_tedi_unet_lunar_lander
```

### Option 2: Manual Setup

#### Step 1: Install Dependencies

```bash
pip install gymnasium[box2d] stable-baselines3 huggingface-sb3 zarr numcodecs tqdm
```

#### Step 2: Collect Demonstration Data

```bash
# Collect 200 successful episodes (takes ~10-20 minutes)
python scripts/collect_lunar_lander_demos.py \
    --n_episodes 200 \
    --output data/lunar_lander/demonstrations.zarr \
    --min_reward 200
```

**Data Collection Options:**
- `--n_episodes`: Number of successful episodes to collect (default: 200)
- `--min_reward`: Minimum reward threshold (200 = solved, default: 200)
- `--discrete`: Use discrete action space instead of continuous
- `--no_pretrained`: Use random policy instead of pre-trained model

#### Step 3: Train SDP

```bash
python train.py \
    --config-dir=diffusion_policy/config/ \
    --config-name=train_tedi_unet_lunar_lander
```

Training will:
- Run for 500 epochs (~2-3 hours on GPU)
- Evaluate every 50 epochs on 20 test episodes
- Save checkpoints and videos to `data/outputs/`
- Log metrics to Weights & Biases

## Files Created

```
streaming_diffusion_policy/
├── scripts/
│   └── collect_lunar_lander_demos.py      # Data collection script
├── diffusion_policy/
│   ├── dataset/
│   │   └── lunar_lander_dataset.py        # Dataset class
│   ├── env_runner/
│   │   └── lunar_lander_runner.py         # Evaluation runner
│   └── config/
│       ├── task/
│       │   └── lunar_lander_lowdim.yaml   # Task configuration
│       └── train_tedi_unet_lunar_lander.yaml  # Training config
├── setup_lunar_lander.sh                   # Automated setup script
└── LUNAR_LANDER_README.md                  # This file
```

## Configuration Details

### Environment
- **Observation space**: 8D continuous
  - `[x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]`
- **Action space**: 2D continuous
  - `[main_engine_thrust, lateral_engine_thrust]`
  - Main engine: [0, 1], Lateral: [-1, 1]
- **Solved threshold**: Reward ≥ 200

### SDP Hyperparameters
- **Horizon**: 16 (predict 16 actions ahead)
- **Observation steps**: 2 (use last 2 observations)
- **Action steps**: 8 (execute first 8 actions)
- **Chunks**: 2 (16 / 8 = 2x speedup)
- **Diffusion steps**: 100 training, 100 inference
- **Batch size**: 256
- **Learning rate**: 1e-4 with cosine schedule

### Training Details
- **Dataset**: ~200 successful episodes (~40k steps)
- **Train/Val split**: 90% / 10%
- **Epochs**: 500
- **EMA**: Yes (momentum 0.75)
- **Evaluation**: Every 50 epochs on 20 test episodes

## Expected Performance

### Data Collection
- **Success rate**: 80-95% (with pre-trained model)
- **Time**: ~10-20 minutes for 200 episodes
- **Episode length**: ~150-250 steps per episode

### Training
- **Time**: 2-3 hours on single GPU (RTX 4090 / A100)
- **Convergence**: ~200-300 epochs
- **Final performance**: 
  - Test mean score: 200-250 (solved = 200)
  - Success rate: 80-90%

## Monitoring Training

Training logs are sent to Weights & Biases. Key metrics:

- `train_loss`: Training loss (should decrease)
- `test_mean_score`: Average reward on test episodes (higher is better)
- `mean_prediction_time`: Inference time per step (should be ~0.05-0.1s)
- Videos: Watch policy performance on test episodes

## Troubleshooting

### "No module named gymnasium"
```bash
pip install gymnasium[box2d]
```

### "Failed to load PPO model"
The script will fall back to random policy. To fix:
```bash
pip install stable-baselines3 huggingface-sb3
```

### "Not enough successful episodes"
Lower the reward threshold:
```bash
python scripts/collect_lunar_lander_demos.py --min_reward 100
```

### CUDA out of memory
Reduce batch size in config:
```yaml
dataloader:
  batch_size: 128  # Instead of 256
```

### Training is slow
- Make sure you're using GPU: `training.device: "cuda:0"`
- Check GPU utilization: `nvidia-smi`
- Reduce `num_workers` if CPU is bottleneck

## Next Steps: MAV-EI Integration

Once SDP is trained, you can use it as the fast proposal generator for your MAV-EI project:

### 1. Generate Multiple Proposals

Modify `predict_action()` to sample K candidates:

```python
def predict_action_samples(self, obs_dict, K=10):
    """Sample K action trajectories for verifier selection"""
    samples = []
    for _ in range(K):
        result = self.predict_action(obs_dict)
        samples.append(result['action'])
        self.reset_buffer()  # Reset for independent samples
    return torch.stack(samples, dim=0)  # [K, B, Ta, Da]
```

### 2. Implement Best-of-N Baseline

```python
# Generate proposals
proposals = policy.predict_action_samples(obs, K=10)  # [10, 1, 8, 2]

# Your verifiers evaluate each (placeholder)
scores = [verifier_zoo.evaluate(prop) for prop in proposals]

# Select best
best_idx = np.argmax(scores)
best_action = proposals[best_idx]
```

### 3. Add Your Verifiers

Implement the verifier zoo from your proposal:
- V1: VLM instruction adherence
- V2: Physics feasibility  
- V3: IK reachability
- V4: Affordance critic

Each verifier scores the full action trajectory (not just immediate action).

## Citation

If you use this code, please cite:

```bibtex
@misc{høeg2024streamingdiffusionpolicyfast,
    title={Streaming Diffusion Policy: Fast Policy Synthesis with Variable Noise Diffusion Models}, 
    author={Sigmund H. Høeg and Yilun Du and Olav Egeland},
    year={2024},
    eprint={2406.04806},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2406.04806}, 
}
```

## Questions?

- Check the [original SDP repo](https://github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy)
- Check the [Diffusion Policy repo](https://github.com/real-stanford/diffusion_policy)
- Open an issue on GitHub



