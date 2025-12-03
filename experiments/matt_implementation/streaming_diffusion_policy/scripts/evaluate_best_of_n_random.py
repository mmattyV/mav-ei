#!/usr/bin/env python3
"""
Best-of-N Random Baseline for MAV-EI.

This implements the Week 1 baseline:
- Sample K action candidates from SDP
- Pick one at RANDOM (no verifier)
- Execute that action

This establishes the control baseline to show that verifiers 
actually provide value (random selection shouldn't help).
"""

import os
import sys
import argparse
import random
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


def sample_k_candidates_batched(
    policy: TEDiUnetLowdimPolicy,
    obs_dict: dict,
    k: int,
    device: str = "mps"
) -> list:
    """
    Sample K action candidates from the SDP policy using BATCHED inference.
    
    Instead of calling predict_action K times (slow), we:
    1. Expand the observation to batch size K
    2. Reset the policy buffer to get K independent noise samples
    3. Run ONE forward pass to get K candidates
    
    This is ~K times faster than sequential sampling!
    
    Args:
        policy: The SDP policy
        obs_dict: Dictionary with 'obs' tensor of shape (1, T, D)
        k: Number of candidates to sample
        device: Device to use
        
    Returns:
        List of K action sequences, each shape (n_action_steps, action_dim)
    """
    # Expand observation from (1, T, D) to (K, T, D)
    obs_expanded = obs_dict['obs'].expand(k, -1, -1)  # (K, n_obs_steps, obs_dim)
    obs_dict_batched = {'obs': obs_expanded}
    
    # Force buffer reinitialization to get K independent random noise samples
    # This is critical - without this, all K samples would be identical!
    policy.action_buffer = None
    
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict_batched)
    
    # actions shape: (K, n_action_steps, action_dim)
    actions = action_dict['action'].cpu().numpy()
    
    # Convert to list of K candidates
    candidates = [actions[i] for i in range(k)]
    
    return candidates


def evaluate_best_of_n_random(
    policy: TEDiUnetLowdimPolicy,
    cfg,
    k: int = 5,
    n_episodes: int = 20,
    max_steps: int = 1000,
    device: str = "mps",
    render: bool = False,
    seed: int = 42,
):
    """
    Evaluate Best-of-N Random baseline on discrete LunarLander-v3.
    
    For each decision point:
    1. Generate K action candidates from SDP
    2. Pick one at RANDOM (no scoring/verification)
    3. Execute that action
    
    Args:
        policy: Trained SDP policy
        cfg: Config from checkpoint
        k: Number of candidates to sample (the "N" in Best-of-N)
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        device: Device for inference
        render: Whether to render environment
        seed: Random seed for reproducibility
    """
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    
    print(f"\n{'='*60}")
    print("Best-of-N Random Baseline Evaluation")
    print(f"{'='*60}")
    print(f"  K (candidates per step): {k}")
    print(f"  Selection method: RANDOM (no verifier)")
    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  n_episodes: {n_episodes}")
    print()
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    # Track how many times we actually sample K candidates
    total_sampling_points = 0
    
    pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode")
    
    for ep in pbar:
        obs, info = env.reset(seed=seed + ep)
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        while not done and steps < max_steps:
            if len(action_queue) > 0:
                # Still executing previous action sequence
                action = action_queue.pop(0)
            else:
                # Need new actions - this is where Best-of-N happens
                total_sampling_points += 1
                
                # Stack observations
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_dict = {'obs': obs_tensor}
                
                # Sample K candidates from SDP (batched for speed!)
                candidates = sample_k_candidates_batched(policy, obs_dict, k, device)
                
                # RANDOM SELECTION: Pick one candidate at random
                selected_idx = random.randint(0, k - 1)
                selected_actions = candidates[selected_idx]
                
                # Convert one-hot to discrete actions
                discrete_actions = np.argmax(selected_actions, axis=1)
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
        'k': k,
        'selection_method': 'random',
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes),
        'n_successes': sum(successes),
        'n_episodes': n_episodes,
        'total_sampling_points': total_sampling_points,
        'avg_sampling_per_episode': total_sampling_points / n_episodes,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Best-of-N Random Baseline for MAV-EI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with K=5 candidates
  python evaluate_best_of_n_random.py --checkpoint path/to/checkpoint.ckpt --k 5
  
  # Compare different K values
  python evaluate_best_of_n_random.py --checkpoint path/to/checkpoint.ckpt --k 1   # Baseline (same as single SDP)
  python evaluate_best_of_n_random.py --checkpoint path/to/checkpoint.ckpt --k 5
  python evaluate_best_of_n_random.py --checkpoint path/to/checkpoint.ckpt --k 10
  python evaluate_best_of_n_random.py --checkpoint path/to/checkpoint.ckpt --k 20
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SDP checkpoint file')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of candidates to sample (default: 5)')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (mps/cpu) (default: mps)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAV-EI: Best-of-N Random Baseline")
    print("=" * 60)
    print()
    print("This baseline samples K candidates and picks one at RANDOM.")
    print("Expected: Performance should be ~same as single SDP (K=1)")
    print("This proves that verifiers (not sampling) provide the value.")
    print()
    
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    results = evaluate_best_of_n_random(
        policy=policy,
        cfg=cfg,
        k=args.k,
        n_episodes=args.n_episodes,
        device=args.device,
        render=args.render,
        seed=args.seed,
    )
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  K (candidates):    {results['k']}")
    print(f"  Selection:         {results['selection_method'].upper()}")
    print(f"  Mean Reward:       {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min/Max:           {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean Length:       {results['mean_length']:.1f} steps")
    print(f"  Success Rate:      {results['success_rate']*100:.1f}% ({results['n_successes']}/{results['n_episodes']})")
    print(f"  Sampling Points:   {results['total_sampling_points']} total ({results['avg_sampling_per_episode']:.1f}/episode)")
    print()
    
    # Interpretation
    print("Interpretation:")
    print("-" * 40)
    if results['k'] == 1:
        print("  K=1 is equivalent to single SDP (baseline).")
        print("  Use this as reference for comparing K>1.")
    else:
        print(f"  With K={results['k']} candidates selected RANDOMLY:")
        print("  - If similar to K=1: Random selection doesn't help (expected)")
        print("  - If better: Sampling diversity helps (unexpected)")
        print("  - If worse: Random hurts consistency (possible)")
    print()
    print("Next step: Implement verifier-based selection to beat this baseline!")


if __name__ == "__main__":
    main()

