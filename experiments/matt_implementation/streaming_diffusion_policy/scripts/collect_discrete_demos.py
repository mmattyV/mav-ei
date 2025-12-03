#!/usr/bin/env python3
"""
Collect expert demonstrations for discrete LunarLander-v3.

Collects RAW observations (not normalized) so SDP sees the same
distribution during training and evaluation.
"""

import os
import argparse
import numpy as np
import zarr
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm


def collect_discrete_demos(
    model_path: str,
    normalize_path: str,
    n_episodes: int = 200,
    min_reward: float = 200,
    output_path: str = "data/lunar_lander_discrete/demonstrations.zarr",
):
    """
    Collect demonstrations with RAW observations.
    
    Uses the trained model (which needs normalized obs) but stores
    raw observations for SDP training.
    """
    print("=" * 60)
    print("Collecting Discrete Lunar Lander Demonstrations")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Min reward: {min_reward}")
    print()
    
    # Load model with normalization (for predictions)
    model = PPO.load(model_path)
    model_env = make_vec_env("LunarLander-v3", n_envs=1, seed=42)
    
    if os.path.exists(normalize_path):
        model_env = VecNormalize.load(normalize_path, model_env)
        model_env.training = False
        model_env.norm_reward = False
        print("✓ Loaded model with normalization")
    else:
        print("⚠️  No normalization found")
    
    # Create RAW environment for collecting observations
    raw_env = gym.make("LunarLander-v3")
    
    all_obs = []
    all_actions = []
    episode_ends = []
    total_steps = 0
    collected = 0
    attempted = 0
    
    pbar = tqdm(total=n_episodes, desc="Collecting")
    
    while collected < n_episodes:
        seed = 42 + attempted
        
        # Reset both environments
        raw_obs, _ = raw_env.reset(seed=seed)
        model_env.seed(seed)
        model_obs = model_env.reset()
        
        done = False
        ep_obs = []
        ep_actions = []
        total_reward = 0
        
        while not done:
            # Get action from model (needs normalized obs)
            action, _ = model.predict(model_obs, deterministic=True)
            
            # Store RAW observation and action
            ep_obs.append(raw_obs.copy())
            
            # For discrete: action is an integer (0-3)
            # Convert to one-hot for SDP
            action_onehot = np.zeros(4, dtype=np.float32)
            action_onehot[action[0]] = 1.0
            ep_actions.append(action_onehot)
            
            # Step both environments
            raw_obs, reward, terminated, truncated, _ = raw_env.step(action[0])
            model_obs, _, model_done, _ = model_env.step(action)
            
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
    raw_env.close()
    model_env.close()
    
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
    print(f"  Action shape: {action_array.shape} (one-hot encoded)")
    print(f"  Obs range: [{obs_array.min():.2f}, {obs_array.max():.2f}]")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Collect discrete Lunar Lander demos')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PPO model')
    parser.add_argument('--normalize', type=str, default=None,
                        help='Path to VecNormalize stats (default: same dir as model)')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes to collect')
    parser.add_argument('--min_reward', type=float, default=200,
                        help='Minimum reward to keep episode')
    parser.add_argument('--output', type=str, 
                        default='data/lunar_lander_discrete/demonstrations.zarr',
                        help='Output zarr path')
    
    args = parser.parse_args()
    
    # Default normalize path
    if args.normalize is None:
        args.normalize = os.path.join(os.path.dirname(args.model), "vec_normalize.pkl")
    
    collect_discrete_demos(
        model_path=args.model,
        normalize_path=args.normalize,
        n_episodes=args.episodes,
        min_reward=args.min_reward,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

