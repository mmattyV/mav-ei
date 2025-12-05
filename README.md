# MAV-EI: Multi-Agent Verifier Selection for Embodied Intelligence

A research project exploring test-time verification strategies for improving action proposals from fast policy generators in embodied control tasks.

## Abstract

Inspired by the success of multi-agent verification in improving LLM outputs, MAV-EI implements a "society of verifiers" that evaluates and steers action proposals from fast policy generators. Unlike approaches that only scale the number of sampled candidates with a single verifier, MAV-EI scales with both the number of candidates and diversity of verifiers (image-based and state-based).

## Key Results

Our ablation study on Lunar Lander shows that hybrid verifiers (image + state) significantly improve success rates over random selection:

| Configuration | Success Rate | Mean Reward |
|--------------|--------------|-------------|
| Random Baseline | ~48% | ~100 |
| Image-Only Verifiers | ~52% | ~120 |
| State-Only Verifiers | ~68% | ~180 |
| Hybrid (Equal Weights) | ~70% | ~190 |
| **Hybrid (Tuned Weights)** | **~74%** | **~200** |

See `streaming_diffusion_policy/results/` for full results and plots.

## Repository Structure

```
mav-ei/
├── expert_dqn/                      # DQN expert for demo collection
│   ├── main.py                      # Train/watch/benchmark DQN
│   ├── dqn_agent.py                 # DQN implementation
│   └── lander_dqn_good.pth          # Pre-trained expert weights
│
├── streaming_diffusion_policy/      # SDP + MAV-EI verifiers
│   ├── train.py                     # SDP training entry point
│   ├── scripts/
│   │   ├── collect_dqn_demos.py     # Collect demos from DQN expert
│   │   └── run_ablation_study.py    # Run full ablation study
│   ├── verifiers/                   # Image & state-based verifiers
│   ├── diffusion_policy/            # Core SDP code + Lunar Lander adapters
│   ├── results/                     # Ablation study outputs
│   └── data/                        # Demos & checkpoints (gitignored)
│
└── .gitignore
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f streaming_diffusion_policy/conda_environment_sdp_lunar_lander.yaml
conda activate sdp_lunar_lander
```

### 2. Collect Expert Demonstrations

```bash
cd streaming_diffusion_policy

# Collect 200 successful episodes from DQN expert
python scripts/collect_dqn_demos.py \
  --weights ../expert_dqn/lander_dqn_good.pth \
  --episodes 200 \
  --output data/lunar_lander_dqn/demonstrations.zarr
```

### 3. Train Streaming Diffusion Policy

```bash
# Train SDP (50 epochs, ~30 min on Apple Silicon)
python -W ignore train.py --config-name=train_tedi_unet_lunar_lander
```

Checkpoints saved to: `data/outputs/{date}/{time}_.../checkpoints/`

### 4. Run Ablation Study

```bash
# Full ablation (5 experiments × 50 episodes)
python -W ignore scripts/run_ablation_study.py \
  --checkpoint data/outputs/past_checkpoints/epoch=0025-train_loss=0.044639.ckpt \
  --n_episodes 50

# Quick test (2 experiments × 10 episodes)
python -W ignore scripts/run_ablation_study.py \
  --checkpoint data/outputs/past_checkpoints/epoch=0025-train_loss=0.044639.ckpt \
  --n_episodes 10 \
  --experiments random_baseline hybrid_tuned
```

Results saved to: `results/`

## Verifiers

### Image-Based Verifiers
- **Vertical Corridor**: Is the lander within the landing zone?
- **Controlled Descent**: Is the main engine firing with stable orientation?
- **Pitch Trimming**: Are side thrusters correcting tilt?

### State-Based Verifiers
- **Safe Velocity**: Horizontal/vertical velocity within safe bounds
- **Stable Rotation**: Angular velocity controlled
- **On Target**: Predicted trajectory leads to landing zone
- **Safe Landing Condition**: Combined check for final approach
- **Descending**: Lander moving downward (not climbing)
- **In Corridor**: X-position within landing zone bounds

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  DQN Expert     │────▶│  Demonstrations  │────▶│  Train SDP      │
│  (expert_dqn/)  │     │  (zarr format)   │     │  (train.py)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Results &      │◀────│  Ablation Study  │◀────│  SDP Checkpoint │
│  Analysis       │     │  (verifiers)     │     │  (.ckpt)        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Citation

If you use this code, please cite:

```bibtex
@misc{mav-ei2024,
  title={MAV-EI: Multi-Agent Verifier Selection for Embodied Intelligence},
  author={Matthew Vu},
  year={2024},
}
```

## Acknowledgements

This project builds upon:

- **[Streaming Diffusion Policy](https://github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy)** by Høeg et al. - Fast policy synthesis with variable noise diffusion models. Licensed under MIT.
  
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

- **[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)** - The foundation for SDP.

- **[Gymnasium](https://gymnasium.farama.org/)** - Lunar Lander environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The Streaming Diffusion Policy code in `streaming_diffusion_policy/` is also MIT licensed by its original authors.
