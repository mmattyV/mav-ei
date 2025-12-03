#!/usr/bin/env python3
"""
Evaluate trained SDP on discrete LunarLander-v3.
"""

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy


def load_checkpoint(checkpoint_path: str, device: str = "mps"):
    """Load a trained SDP checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['cfg']
    
    policy: TEDiUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
    
    # Use EMA weights if available
    if 'state_dicts' in checkpoint and 'ema_model' in checkpoint['state_dicts']:
        print("Using EMA weights")
        policy.load_state_dict(checkpoint['state_dicts']['ema_model'])
    else:
        print("Using regular weights")
        policy.load_state_dict(checkpoint['state_dicts']['model'])
    
    policy.to(device)
    policy.eval()
    
    return policy, cfg


def evaluate_policy(
    policy: TEDiUnetLowdimPolicy,
    cfg,
    n_episodes: int = 20,
    max_steps: int = 1000,
    device: str = "mps",
    render: bool = False,
):
    """Evaluate the policy on discrete LunarLander-v3."""
    
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    
    print(f"\nEvaluation Config:")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  action_dim: 4 (discrete one-hot)")
    print()
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode")
    
    for ep in pbar:
        obs, info = env.reset()
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        while not done and steps < max_steps:
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                # Stack observations
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_dict = {'obs': obs_tensor}
                
                # Get action from policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                # actions shape: (1, n_action_steps, 4)
                actions = action_dict['action'].cpu().numpy()[0]
                
                # Convert one-hot to discrete actions
                discrete_actions = np.argmax(actions, axis=1)
                action_queue = list(discrete_actions)
                action = action_queue.pop(0)
            
            # Step environment with discrete action
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            obs_history.append(next_obs)
            total_reward += reward
            steps += 1
        
        success = total_reward >= 200
        successes.append(success)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        current_mean = np.mean(episode_rewards)
        current_success = np.mean(successes) * 100
        pbar.set_postfix({
            'mean_reward': f'{current_mean:.1f}',
            'success': f'{current_success:.0f}%',
            'last': f'{total_reward:.1f}'
        })
    
    pbar.close()
    env.close()
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes),
        'n_successes': sum(successes),
        'n_episodes': n_episodes,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SDP on discrete Lunar Lander')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (mps/cpu)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDP Discrete Lunar Lander Evaluation")
    print("=" * 60)
    
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    results = evaluate_policy(
        policy=policy,
        cfg=cfg,
        n_episodes=args.n_episodes,
        device=args.device,
        render=args.render,
    )
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min/Max:       {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean Length:   {results['mean_length']:.1f} steps")
    print(f"  Success Rate:  {results['success_rate']*100:.1f}% ({results['n_successes']}/{results['n_episodes']})")
    print()
    
    if results['mean_reward'] >= 200:
        print("🎉 Model SOLVED the environment!")
    elif results['mean_reward'] >= 100:
        print("📈 Model is learning but needs more training")
    else:
        print("⚠️  Model needs significant improvement")


if __name__ == "__main__":
    main()

