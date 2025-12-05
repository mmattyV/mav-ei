# Streaming Diffusion Policy + MAV-EI Verifiers

This directory contains the Streaming Diffusion Policy (SDP) codebase adapted for the Lunar Lander task, with MAV-EI verifier-based action selection.

## Original Work

This code is based on **[Streaming Diffusion Policy](https://github.com/Streaming-Diffusion-Policy/streaming_diffusion_policy)** by Høeg et al., which builds upon [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

**Original SDP contributions:**
- State- and image-based SDP in `diffusion_policy/policy/tedi_unet_lowdim_policy.py` and `tedi_unet_hybrid_image_policy.py`
- Noise scheduler in `diffusion_policy/policy/schedulers.py`
- Modified model architecture in `diffusion_policy/model/diffusion/conditional_unet1d_tedi.py`

If you use this code, please cite the original SDP paper:

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

## MAV-EI Additions

Our contributions for the MAV-EI project:

| File/Folder | Description |
|-------------|-------------|
| `diffusion_policy/dataset/lunar_lander_dataset.py` | Lunar Lander dataset adapter |
| `diffusion_policy/config/task/lunar_lander_lowdim.yaml` | Task configuration |
| `diffusion_policy/config/train_tedi_unet_lunar_lander.yaml` | Training configuration |
| `verifiers/` | Image and state-based verifiers |
| `scripts/collect_dqn_demos.py` | Demo collection from DQN expert |
| `scripts/run_ablation_study.py` | Full ablation study with plots |
| `results/` | Ablation study outputs |

## Installation

```bash
conda env create -f conda_environment_sdp_lunar_lander.yaml
conda activate sdp_lunar_lander
```

## Quick Start

### 1. Collect Demonstrations

```bash
python scripts/collect_dqn_demos.py \
  --weights ../expert_dqn/lander_dqn_good.pth \
  --episodes 200 \
  --output data/lunar_lander_dqn/demonstrations.zarr
```

### 2. Train SDP

```bash
python -W ignore train.py --config-name=train_tedi_unet_lunar_lander
```

**Training options:**
```bash
# Quick test (3 epochs)
python -W ignore train.py --config-name=train_tedi_unet_lunar_lander training.num_epochs=3

# Full training (50 epochs, ~30 min on Apple Silicon)
python -W ignore train.py --config-name=train_tedi_unet_lunar_lander
```

Checkpoints saved to: `data/outputs/{date}/{time}_train_tedi_unet_lowdim_lunar_lander_lowdim/checkpoints/`

### 3. Run Ablation Study

```bash
# Full study (all 5 experiments)
python -W ignore scripts/run_ablation_study.py \
  --checkpoint data/outputs/past_checkpoints/epoch=0025-train_loss=0.044639.ckpt \
  --n_episodes 50

# Quick test
python -W ignore scripts/run_ablation_study.py \
  --checkpoint data/outputs/past_checkpoints/epoch=0025-train_loss=0.044639.ckpt \
  --n_episodes 10 \
  --experiments random_baseline hybrid_tuned
```

**Ablation experiments:**
1. `random_baseline` - Random selection (no verifiers)
2. `image_only` - Image-based verifiers only
3. `state_only` - State-based verifiers only
4. `hybrid_equal` - All verifiers, equal weights
5. `hybrid_tuned` - All verifiers, tuned weights

## Results

Results are saved to `results/`:
- `ablation_results.json` - Raw data
- `success_rate_comparison.png` - Bar chart
- `reward_distribution.png` - Box plot
- `verifier_analysis.png` - Pass rates by success/failure
- `timing_comparison.png` - Computational cost
- `latex_tables.txt` - Copy-paste LaTeX

## Verifiers

### Image-Based (`verifiers/`)
| Verifier | Description |
|----------|-------------|
| `VerticalCorridorVerifier` | Checks if lander is within landing zone |
| `ControlledDescentVerifier` | Checks if main engine is firing with stable orientation |
| `PitchTrimmingVerifier` | Checks if side thrusters are correcting tilt |

### State-Based (`verifiers/state_based.py`)
| Verifier | Description |
|----------|-------------|
| `SafeVelocityVerifier` | Velocity within safe bounds |
| `StableRotationVerifier` | Angular velocity controlled |
| `OnTargetVerifier` | Predicted trajectory leads to landing zone |
| `SafeLandingConditionVerifier` | Combined final approach check |
| `DescendingVerifier` | Lander moving downward |
| `InCorridorVerifier` | X-position within landing zone |

## Directory Structure

```
streaming_diffusion_policy/
├── train.py                         # Training entry point
├── conda_environment_sdp_lunar_lander.yaml
├── scripts/
│   ├── collect_dqn_demos.py         # Demo collection
│   └── run_ablation_study.py        # Ablation study
├── verifiers/
│   ├── vertical_corridor.py
│   ├── controlled_descent.py
│   ├── pitch_trimming.py
│   └── state_based.py
├── diffusion_policy/
│   ├── config/
│   │   ├── task/lunar_lander_lowdim.yaml
│   │   └── train_tedi_unet_lunar_lander.yaml
│   ├── dataset/lunar_lander_dataset.py
│   └── ...                          # Original SDP code
├── results/                         # Ablation outputs
└── data/                            # Demos & checkpoints (gitignored)
```

## License

MIT License - see [LICENSE](LICENSE).

Original Streaming Diffusion Policy code is also MIT licensed by Høeg et al.
