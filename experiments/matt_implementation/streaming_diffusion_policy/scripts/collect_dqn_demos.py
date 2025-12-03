#!/usr/bin/env python3
"""
Collect expert demonstrations using Ryan's pre-trained DQN.

Uses the QNetwork architecture from ryan_implementation/dqn_agent.py
with weights from lander_dqn_good.pth
"""

import os
import argparse
import numpy as np
import zarr
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm


# Copy of QNetwork from ryan_implementation/dqn_agent.py
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_dqn_model(weights_path: str, device: str = "cpu"):
    """Load the pre-trained DQN model."""
    # LunarLander-v3: 8 obs dims, 4 actions
    model = QNetwork(state_dim=8, action_dim=4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def select_action(model: QNetwork, state: np.ndarray, device: str = "cpu") -> int:
    """Select action using greedy policy."""
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_t)
    return int(torch.argmax(q_values, dim=1).item())


def collect_dqn_demos(
    weights_path: str,
    n_episodes: int = 200,
    min_reward: float = 200,
    output_path: str = "data/lunar_lander_dqn/demonstrations.zarr",
    device: str = "cpu",
):
    """
    Collect demonstrations using Ryan's DQN model.
    
    Stores RAW observations (no normalization needed - DQN was trained on raw obs).
    """
    print("=" * 60)
    print("Collecting Demos with Ryan's DQN")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Min reward: {min_reward}")
    print()
    
    # Load model
    model = load_dqn_model(weights_path, device)
    print("✓ Loaded DQN model")
    
    # Create environment
    env = gym.make("LunarLander-v3")
    
    all_obs = []
    all_actions = []
    episode_ends = []
    total_steps = 0
    collected = 0
    attempted = 0
    
    pbar = tqdm(total=n_episodes, desc="Collecting")
    
    while collected < n_episodes:
        obs, _ = env.reset(seed=42 + attempted)
        done = False
        ep_obs = []
        ep_actions = []
        total_reward = 0
        
        while not done:
            action = select_action(model, obs, device)
            
            # Store observation and one-hot action
            ep_obs.append(obs.copy())
            action_onehot = np.zeros(4, dtype=np.float32)
            action_onehot[action] = 1.0
            ep_actions.append(action_onehot)
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        attempted += 1
        
        if total_reward >= min_reward:
            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            total_steps += len(ep_obs)
            episode_ends.append(total_steps)
            collected += 1
            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{total_reward:.0f}',
                'rate': f'{collected/attempted*100:.0f}%'
            })
    
    pbar.close()
    env.close()
    
    print()
    print(f"Collected {collected} episodes ({collected/attempted*100:.1f}% success rate)")
    print(f"Total steps: {total_steps}")
    
    # Convert to arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    action_array = np.array(all_actions, dtype=np.float32)
    episode_ends_array = np.array(episode_ends, dtype=np.int64)
    
    # Save to zarr
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    root = zarr.open(output_path, mode='w')
    
    data = root.create_group('data')
    data.create_dataset('obs', data=obs_array, chunks=(1000, obs_array.shape[1]))
    data.create_dataset('action', data=action_array, chunks=(1000, action_array.shape[1]))
    
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=episode_ends_array)
    
    print()
    print(f"✓ Saved to {output_path}")
    print(f"  Obs shape: {obs_array.shape}")
    print(f"  Action shape: {action_array.shape}")
    print(f"  Obs range: [{obs_array.min():.2f}, {obs_array.max():.2f}]")
    
    return output_path


def test_dqn(weights_path: str, n_episodes: int = 10, render: bool = False):
    """Test the DQN model."""
    print("Testing DQN model...")
    
    model = load_dqn_model(weights_path)
    
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = select_action(model, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        status = "✓" if total_reward >= 200 else "✗"
        print(f"  Episode {ep+1}: {total_reward:.2f} {status}")
    
    env.close()
    
    print()
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success rate: {sum(r >= 200 for r in rewards)}/{n_episodes}")


def main():
    parser = argparse.ArgumentParser(description='Collect demos with DQN')
    parser.add_argument('--weights', type=str, default='models/lander_dqn_good.pth',
                        help='Path to DQN weights')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to collect')
    parser.add_argument('--min_reward', type=float, default=200,
                        help='Minimum reward to keep episode')
    parser.add_argument('--output', type=str,
                        default='data/lunar_lander_dqn/demonstrations.zarr',
                        help='Output zarr path')
    parser.add_argument('--test', action='store_true',
                        help='Test the model instead of collecting')
    parser.add_argument('--test_render', action='store_true',
                        help='Test with visualization')
    
    args = parser.parse_args()
    
    if args.test or args.test_render:
        test_dqn(args.weights, render=args.test_render)
    else:
        collect_dqn_demos(
            weights_path=args.weights,
            n_episodes=args.episodes,
            min_reward=args.min_reward,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()

