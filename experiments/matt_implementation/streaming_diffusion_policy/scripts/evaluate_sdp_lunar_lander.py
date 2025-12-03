#!/usr/bin/env python3
"""
Evaluate trained Streaming Diffusion Policy on Lunar Lander.

This script loads a trained SDP checkpoint and runs it in the 
LunarLanderContinuous-v3 environment to measure performance.
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
from omegaconf import OmegaConf
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply


def load_checkpoint(checkpoint_path: str, device: str = "mps"):
    """Load a trained SDP checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False needed because checkpoint contains OmegaConf config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    cfg = checkpoint['cfg']
    
    # Instantiate policy
    policy: TEDiUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
    
    # Load EMA weights if available, otherwise use regular weights
    # EMA weights are typically better for inference
    if 'state_dicts' in checkpoint and 'ema_model' in checkpoint['state_dicts']:
        print("Using EMA weights (recommended)")
        ema_state = checkpoint['state_dicts']['ema_model']
        policy.load_state_dict(ema_state)
    elif 'ema_model' in checkpoint:
        print("Using EMA weights")
        policy.load_state_dict(checkpoint['ema_model'])
    else:
        print("Using regular weights (EMA not available)")
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
    verbose: bool = True,
):
    """
    Evaluate the policy on LunarLanderContinuous-v3.
    
    Returns:
        dict with evaluation metrics
    """
    # Create environment
    render_mode = "human" if render else None
    env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
    
    # Get dimensions from config
    n_obs_steps = cfg.n_obs_steps  # How many past observations to use
    n_action_steps = cfg.n_action_steps  # How many actions to execute per prediction
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    
    if verbose:
        print(f"\nEvaluation Config:")
        print(f"  n_obs_steps: {n_obs_steps}")
        print(f"  n_action_steps: {n_action_steps}")
        print(f"  obs_dim: {obs_dim}")
        print(f"  action_dim: {action_dim}")
        print(f"  Episodes: {n_episodes}")
        print()
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    # Progress bar
    pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode")
    
    for ep in pbar:
        obs, info = env.reset()
        
        # Initialize observation history
        obs_history = [obs] * n_obs_steps
        
        # Reset policy buffer
        policy.reset_buffer()
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        while not done and steps < max_steps:
            # If we have actions queued, use them
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                # Need to generate new actions
                # Stack observations: (n_obs_steps, obs_dim)
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                
                # Convert to tensor: (1, n_obs_steps, obs_dim)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                
                # Create observation dict
                obs_dict = {'obs': obs_tensor}
                
                # Get action from policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                # action_dict['action'] shape: (1, n_action_steps, action_dim)
                actions = action_dict['action'].cpu().numpy()[0]  # (n_action_steps, action_dim)
                
                # Queue all actions
                action_queue = list(actions)
                action = action_queue.pop(0)
            
            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update history
            obs_history.append(next_obs)
            
            total_reward += reward
            steps += 1
        
        # Check success (landed safely = reward >= 200)
        success = total_reward >= 200
        successes.append(success)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Update progress bar with running stats
        current_mean = np.mean(episode_rewards)
        current_success = np.mean(successes) * 100
        pbar.set_postfix({
            'mean_reward': f'{current_mean:.1f}',
            'success': f'{current_success:.0f}%',
            'last': f'{total_reward:.1f}'
        })
    
    pbar.close()
    env.close()
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes),
        'n_successes': sum(successes),
        'n_episodes': n_episodes,
        'all_rewards': episode_rewards,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained SDP on Lunar Lander')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.ckpt)')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of episodes to evaluate (default: 20)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (default: mps for Apple Silicon)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (slower)')
    parser.add_argument('--quiet', action='store_true',
                        help='Less verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDP Lunar Lander Evaluation")
    print("=" * 60)
    
    # Load policy
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Evaluate
    results = evaluate_policy(
        policy=policy,
        cfg=cfg,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        device=args.device,
        render=args.render,
        verbose=not args.quiet,
    )
    
    # Print results
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
        print("🎉 Model SOLVED the environment! (mean reward >= 200)")
    elif results['mean_reward'] >= 100:
        print("📈 Model is learning but needs more training (mean reward >= 100)")
    else:
        print("⚠️  Model needs significant improvement (mean reward < 100)")
    
    return results


if __name__ == "__main__":
    main()

