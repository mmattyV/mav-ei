#!/usr/bin/env python3
"""
MAV-EI: Compare Random vs Verifier-Based Selection.

This version uses PARALLEL ENVIRONMENTS to properly simulate and 
screenshot each candidate before selection.

For each decision point:
1. Create K fresh environments
2. Replay action history to sync them to current state
3. Simulate each candidate in its own environment
4. Take screenshots, run image verifiers
5. Pick candidate with best verifier score
6. Execute in real environment
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import gymnasium as gym
import cv2
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy

from verifiers.vertical_corridor import VerticalCorridorVerifier
from verifiers.controlled_descent import ControlledDescentVerifier
from verifiers.pitch_trimming import PitchTrimmingVerifier
# from verifiers.final_success import FinalSuccessVerifier  # Disabled - gives false positives


def load_checkpoint(checkpoint_path: str, device: str = "mps"):
    """Load a trained SDP checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['cfg']
    
    policy: TEDiUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
    
    if 'state_dicts' in checkpoint and 'ema_model' in checkpoint['state_dicts']:
        policy.load_state_dict(checkpoint['state_dicts']['ema_model'])
    else:
        policy.load_state_dict(checkpoint['state_dicts']['model'])
    
    policy.to(device)
    policy.eval()
    
    return policy, cfg


def sample_k_candidates_batched(policy, obs_dict, k, device):
    """Sample K action candidates from SDP."""
    obs_expanded = obs_dict['obs'].expand(k, -1, -1)
    policy.action_buffer = None
    
    with torch.no_grad():
        action_dict = policy.predict_action({'obs': obs_expanded})
    
    actions = action_dict['action'].cpu().numpy()
    return [np.argmax(actions[i], axis=1) for i in range(k)]


class VerifierScorer:
    """Score frames using image verifiers with weighted scoring.
    
    vertical_corridor is weighted 2x since it's the key differentiator for success.
    """
    
    def __init__(self):
        self.verifiers = {
            'vertical_corridor': VerticalCorridorVerifier(),
            'controlled_descent': ControlledDescentVerifier(),
            'pitch_trimming': PitchTrimmingVerifier(),
        }
        # Weights for each verifier (vertical_corridor is most important)
        self.weights = {
            'vertical_corridor': 2,  # 2x weight - key differentiator
            'controlled_descent': 1,
            'pitch_trimming': 1,
        }
        # Final success verifier disabled - gives false positives during lookahead
        # self.final_success_verifier = FinalSuccessVerifier()
    
    def score_frame(self, frame_bgr):
        """
        Score a single BGR frame with weighted verifiers.
        
        Returns:
            (score, is_final_success): score from verifiers, is_final_success always False (disabled)
        """
        score = 0
        for name, verifier in self.verifiers.items():
            if verifier(frame_bgr):
                score += self.weights.get(name, 1)
        
        # Final success verifier disabled - was giving false positives
        is_final_success = False
        
        return score, is_final_success


def create_synced_env(seed: int, action_history: list):
    """
    Create a fresh environment and replay action history to sync it.
    
    Args:
        seed: The seed used for the episode
        action_history: List of actions taken so far
        
    Returns:
        env: Environment synced to current state
        obs: Current observation after replay
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    
    # Replay all actions to get to current state
    for action in action_history:
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break
    
    return env, obs


def evaluate_candidate_parallel(
    seed: int,
    action_history: list,
    candidate_actions: list,
    scorer: VerifierScorer,
    n_lookahead: int = 4,
):
    """
    Evaluate a candidate by creating a fresh env, replaying history,
    then simulating the candidate's actions and scoring with verifiers.
    
    Args:
        seed: Episode seed
        action_history: Actions taken so far in the real episode
        candidate_actions: The candidate's proposed action sequence
        scorer: VerifierScorer instance
        n_lookahead: How many steps to simulate
        
    Returns:
        (score, achieved_final_success): Total verifier score and whether final success was achieved
    """
    # Create fresh env synced to current state
    env, obs = create_synced_env(seed, action_history)
    
    total_score = 0
    achieved_final_success = False
    
    # Simulate candidate's actions and score
    for i, action in enumerate(candidate_actions[:n_lookahead]):
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        # Capture and score frame
        frame_rgb = env.render()
        if frame_rgb is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            verifier_score, is_final_success = scorer.score_frame(frame_bgr)
            total_score += verifier_score
            
            # If we achieved final success, this candidate is a winner!
            if is_final_success:
                achieved_final_success = True
                break
        
        if terminated or truncated:
            break
    
    env.close()
    
    return total_score, achieved_final_success


def run_evaluation(policy, cfg, k, selection_method, n_episodes, n_lookahead, device, seed):
    """Run evaluation with specified selection method."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    n_obs_steps = cfg.n_obs_steps
    
    scorer = VerifierScorer() if selection_method == 'verifier' else None
    
    episode_rewards = []
    successes = []
    
    desc = f"{'Verifier' if selection_method == 'verifier' else 'Random'} K={k}"
    pbar = tqdm(range(n_episodes), desc=desc, unit="ep")
    
    for ep in pbar:
        episode_seed = seed + ep
        obs, info = env.reset(seed=episode_seed)
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        # Track action history for replaying in parallel envs
        action_history = []
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        while not done and steps < 1000:
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                # Decision point
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                
                candidates = sample_k_candidates_batched(policy, {'obs': obs_tensor}, k, device)
                
                if selection_method == 'random':
                    selected_idx = random.randint(0, k - 1)
                else:  # verifier
                    # Evaluate each candidate in parallel environments
                    scores = []
                    
                    for i, candidate in enumerate(candidates):
                        score, _ = evaluate_candidate_parallel(
                            seed=episode_seed,
                            action_history=action_history,
                            candidate_actions=candidate,
                            scorer=scorer,
                            n_lookahead=n_lookahead,
                        )
                        scores.append(score)
                    
                    # Pick highest score (no early exit - FinalSuccess can give false positives)
                    selected_idx = np.argmax(scores)
                
                action_queue = list(candidates[selected_idx])
                action = action_queue.pop(0)
            
            # Execute action in real environment
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            action_history.append(action)  # Track for parallel env replay
            
            done = terminated or truncated
            obs_history.append(next_obs)
            total_reward += reward
            steps += 1
        
        success = total_reward >= 200
        successes.append(success)
        episode_rewards.append(total_reward)
        
        pbar.set_postfix({'sr': f'{np.mean(successes)*100:.0f}%', 'reward': f'{total_reward:.0f}'})
    
    pbar.close()
    env.close()
    
    return {
        'method': selection_method,
        'k': k,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(successes),
        'n_successes': sum(successes),
        'n_episodes': n_episodes,
    }


def print_comparison(results_random, results_verifier):
    """Print comparison table."""
    print()
    print("=" * 70)
    print("COMPARISON: Random vs Verifier-Based Selection")
    print("=" * 70)
    print()
    print(f"{'Method':<25} {'K':>4} {'Success Rate':>14} {'Mean Reward':>16}")
    print("-" * 70)
    
    sr_random = results_random['success_rate'] * 100
    sr_verifier = results_verifier['success_rate'] * 100
    
    print(f"{'Random Selection':<25} {results_random['k']:>4} "
          f"{sr_random:>10.1f}% ({results_random['n_successes']}/{results_random['n_episodes']}) "
          f"{results_random['mean_reward']:>10.1f} ± {results_random['std_reward']:.1f}")
    
    print(f"{'Verifier Selection':<25} {results_verifier['k']:>4} "
          f"{sr_verifier:>10.1f}% ({results_verifier['n_successes']}/{results_verifier['n_episodes']}) "
          f"{results_verifier['mean_reward']:>10.1f} ± {results_verifier['std_reward']:.1f}")
    
    print("-" * 70)
    
    sr_diff = sr_verifier - sr_random
    reward_diff = results_verifier['mean_reward'] - results_random['mean_reward']
    
    if sr_diff > 0:
        print(f"{'IMPROVEMENT':<25} {'':<4} {'+' + f'{sr_diff:.1f}%':>14} {'+' + f'{reward_diff:.1f}':>16}")
        print()
        print("✓ Verifier selection BEATS random selection!")
    elif sr_diff < 0:
        print(f"{'DIFFERENCE':<25} {'':<4} {f'{sr_diff:.1f}%':>14} {f'{reward_diff:.1f}':>16}")
        print()
        print("✗ Random selection performed better")
    else:
        print(f"{'DIFFERENCE':<25} {'':<4} {'0.0%':>14} {f'{reward_diff:.1f}':>16}")
        print()
        print("= Both methods performed equally")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare Random vs Verifier Selection')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--lookahead', type=int, default=4)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (>1.0 = more diverse candidates)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MAV-EI: Selection Method Comparison (Parallel Environments)")
    print("=" * 70)
    print()
    print(f"  K (candidates):      {args.k}")
    print(f"  Lookahead steps:     {args.lookahead}")
    print(f"  Episodes per method: {args.n_episodes}")
    print(f"  Temperature:         {args.temperature}")
    print()
    print("  Strategy: For each candidate, create fresh env, replay history,")
    print("            simulate lookahead steps, screenshot, run verifiers.")
    print()
    
    # Load policy
    print("Loading SDP policy...")
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Set temperature for sampling diversity
    if args.temperature != 1.0:
        policy.set_temperature(args.temperature)
        print(f"Temperature set to {args.temperature} for more diverse sampling")
    print()
    
    # Run random baseline
    print("=" * 40)
    print("Phase 1: Random Selection Baseline")
    print("=" * 40)
    results_random = run_evaluation(
        policy, cfg, args.k, 'random', 
        args.n_episodes, args.lookahead, args.device, args.seed
    )
    
    print()
    print("=" * 40)
    print("Phase 2: Verifier-Based Selection")
    print("=" * 40)
    print("(This is slower due to parallel environment simulation)")
    results_verifier = run_evaluation(
        policy, cfg, args.k, 'verifier',
        args.n_episodes, args.lookahead, args.device, args.seed
    )
    
    # Print comparison
    print_comparison(results_random, results_verifier)


if __name__ == "__main__":
    main()
