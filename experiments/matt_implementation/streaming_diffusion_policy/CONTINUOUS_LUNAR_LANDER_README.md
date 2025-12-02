# Continuous LunarLander for SDP

This guide shows you how to train **Streaming Diffusion Policy (SDP)** on **continuous** LunarLander-v3 for your MAV-EI research project.

## 🎯 Why Continuous?

For your MAV-EI proposal (Budgeted Multi-Verifier Selection):
- ✅ **SDP is designed for continuous actions** - diffusion models work best with continuous distributions
- ✅ **Your verifiers need continuous trajectories** - physics, IK, affordances all work better with smooth actions
- ✅ **Real robotics is continuous** - joint torques, velocities, positions
- ✅ **Better diversity** - continuous actions provide richer Best-of-N sampling

## 🚀 Quick Start (Automated)

The easiest way is to use the automated setup script:

```bash
conda activate sdp_lunar_lander
./setup_continuous_lunar_lander.sh
```

This will:
1. Train a continuous PPO model (~20-30 min)
2. Collect 200 expert demonstrations
3. Verify the setup

**Total time**: ~30-40 minutes

---

## 🔧 Manual Setup (Step-by-Step)

If you prefer to run each step manually:

### Step 1: Train Continuous PPO Model

```bash
conda activate sdp_lunar_lander
python scripts/train_continuous_lunar_lander.py --timesteps 500000
```

**Training Details:**
- **Time**: ~20-30 min on CPU, ~5-10 min on GPU
- **Environment**: LunarLanderContinuous-v3
- **Algorithm**: PPO with 8 parallel environments
- **Target reward**: 200+ (environment considered "solved")

**Monitor Training:**
```bash
tensorboard --logdir models/lunar_lander_continuous/tensorboard
```

**Output:**
- `models/lunar_lander_continuous/final_model.zip` - Final trained model
- `models/lunar_lander_continuous/best/best_model.zip` - Best checkpoint
- `models/lunar_lander_continuous/checkpoints/` - Intermediate checkpoints

### Step 2: Collect Expert Demonstrations

```bash
python scripts/collect_lunar_lander_demos.py \
    --model models/lunar_lander_continuous/final_model.zip \
    --n_episodes 200 \
    --min_reward 200
```

**Output:**
- `data/lunar_lander/demonstrations.zarr` - Expert demonstrations in Zarr format

**What's collected:**
- 200 successful episodes (reward ≥ 200)
- State observations: `[x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]`
- Continuous actions: `[main_engine, lateral_engine]` in range `[-1, 1]`

### Step 3: Train SDP

```bash
python train.py --config-name train_tedi_unet_lunar_lander
```

**Training Details:**
- **Method**: Streaming Diffusion Policy (TEDi-UNet)
- **Action prediction horizon**: 16 steps
- **Observation history**: 2 steps
- **Training epochs**: 2000 (configurable)

**Monitor SDP Training:**
```bash
tensorboard --logdir data/outputs
```

---

## 📊 Configuration Files

### Task Config: `diffusion_policy/config/task/lunar_lander_lowdim.yaml`

Key parameters:
```yaml
obs_dim: 8       # LunarLander state dimension
action_dim: 2    # [main_engine, lateral_engine]
continuous: true # Use continuous action space
```

### Training Config: `diffusion_policy/config/train_tedi_unet_lunar_lander.yaml`

Key parameters:
```yaml
n_obs_steps: 2           # Observation history
n_action_steps: 8        # Action prediction horizon  
horizon: 16              # Total sequence length
n_epochs: 2000           # Training epochs
batch_size: 256          # Training batch size
```

---

## 🔬 Action Space Details

**LunarLanderContinuous-v3 Action Space:**

```python
action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
```

- **action[0]** - Main engine throttle:
  - `-1.0 to 0.0`: Engine off
  - `0.0 to 1.0`: 50% to 100% throttle

- **action[1]** - Lateral engines:
  - `-1.0 to -0.5`: Fire left engine
  - `-0.5 to 0.5`: Both engines off
  - `0.5 to 1.0`: Fire right engine

---

## 📈 Expected Performance

### PPO Expert Model
- **Mean reward**: 250-300
- **Success rate**: 90-95% of episodes
- **Episode length**: ~100-200 steps

### SDP (After Training)
- **Sample time**: ~10-20ms per action sequence (very fast!)
- **Quality**: Should match or exceed PPO performance
- **Key advantage**: Can generate K diverse proposals for your verifier zoo

---

## 🎓 For Your MAV-EI Research

Once SDP is trained, you can use it for your proposal generator:

```python
# Generate K proposals
proposals = []
for k in range(K):
    action_seq = sdp_policy.sample()
    proposals.append(action_seq)

# Run your verifier zoo
scores = []
for proposal in proposals:
    physics_score = physics_verifier(proposal)
    ik_score = ik_verifier(proposal)
    affordance_score = affordance_critic(proposal)
    vlm_score = vlm_verifier(proposal)
    
    total_score = compute_weighted_score(
        physics_score, ik_score, affordance_score, vlm_score
    )
    scores.append(total_score)

# Select best proposal
best_proposal = proposals[np.argmax(scores)]
```

---

## 🐛 Troubleshooting

### "No continuous models available on HuggingFace"
**Solution**: Train your own model with `scripts/train_continuous_lunar_lander.py`

### "Training is too slow"
**Solutions**:
- Reduce timesteps: `--timesteps 200000` (still gives good results)
- Use GPU if available (10x faster)
- Reduce parallel environments in the training script (uses less RAM)

### "Model doesn't reach 200 reward"
**Solutions**:
- Train longer: `--timesteps 1000000`
- Check tensorboard for training curves
- Adjust hyperparameters in `scripts/train_continuous_lunar_lander.py`

### "Data collection is slow"
**Solutions**:
- Lower min_reward: `--min_reward 150`
- Use the best model: `--model models/lunar_lander_continuous/best/best_model.zip`
- Collect fewer episodes for testing: `--n_episodes 50`

---

## 📁 File Structure

```
streaming_diffusion_policy/
├── scripts/
│   ├── train_continuous_lunar_lander.py     # Train PPO model
│   └── collect_lunar_lander_demos.py        # Collect demonstrations
├── models/
│   └── lunar_lander_continuous/
│       ├── final_model.zip                  # Trained PPO model
│       ├── best/best_model.zip              # Best checkpoint
│       └── checkpoints/                     # Intermediate saves
├── data/
│   └── lunar_lander/
│       └── demonstrations.zarr              # Expert data
├── diffusion_policy/
│   ├── dataset/lunar_lander_dataset.py      # Dataset loader
│   ├── env_runner/lunar_lander_runner.py    # Environment runner
│   └── config/
│       ├── task/lunar_lander_lowdim.yaml    # Task config
│       └── train_tedi_unet_lunar_lander.yaml # Training config
└── setup_continuous_lunar_lander.sh         # Automated setup
```

---

## 🎯 Next Steps

1. ✅ **Week 1 Goal**: Get SDP up and running (this guide)
2. ⏳ **Week 1 Goal**: Implement Best-of-N baseline
3. ⏳ **Week 2 Goal**: Implement verifier zoo
4. ⏳ **Week 3 Goal**: Train MAV-EI full system

---

## 📚 References

- [Streaming Diffusion Policy Paper](https://arxiv.org/abs/2406.04806)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [LunarLander Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

---

## ✨ Summary

**Continuous actions are essential for your MAV-EI research.** This guide provides everything you need to:
1. Train a high-quality continuous expert
2. Collect demonstrations
3. Train SDP for fast proposal generation
4. Build your verifier-based selection system

Questions? Check the troubleshooting section or review the code comments in the scripts!

