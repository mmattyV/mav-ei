# Expert DQN Training

This folder contains code for training a DQN expert agent for the Lunar Lander environment. The trained expert is used to collect demonstrations for training the Streaming Diffusion Policy (SDP).

## Contents

| File | Description |
|------|-------------|
| `main.py` | Main training script with train/watch/benchmark modes |
| `dqn_agent.py` | DQN agent implementation (neural network + replay buffer) |
| `lander_dqn_good.pth` | Pre-trained expert weights (~100% success rate) |

## Usage

### Train a new DQN expert
```bash
cd expert_dqn
python main.py --mode train --train-episodes 1000 --weights lander_dqn_new.pth
```

### Watch the trained expert
```bash
python main.py --mode watch --weights lander_dqn_good.pth --episodes 5
```

### Benchmark the expert
```bash
python main.py --mode benchmark --weights lander_dqn_good.pth --episodes 100
```

Expected output:
```
Success Rate:    100/100 (100.0%)
Average Reward:  282.64 ± 12.93
```

## Dependencies

Requires the `sdp_lunar_lander` conda environment (see main README):

```bash
conda activate sdp_lunar_lander
```

Or install manually:
```bash
pip install gymnasium torch numpy pygame opencv-python
```

## Next Steps

After training/verifying the expert, collect demonstrations for SDP training:

```bash
cd ../streaming_diffusion_policy
python scripts/collect_dqn_demos.py \
  --weights ../expert_dqn/lander_dqn_good.pth \
  --episodes 200
```
