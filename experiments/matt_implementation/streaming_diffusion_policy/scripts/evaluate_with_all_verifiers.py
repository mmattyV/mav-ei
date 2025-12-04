#!/usr/bin/env python3
"""
MAV-EI: Evaluation with Both Image and State-Based Verifiers

This script evaluates verifier-based selection using:
- Image-based verifiers (vertical_corridor, controlled_descent, pitch_trimming)
- State-based verifiers (safe_velocity, stable_rotation, on_target, etc.)

The state-based verifiers are more accurate for dynamic properties like velocity.
"""

import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import gymnasium as gym
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy

# Image-based verifiers
from verifiers.vertical_corridor import VerticalCorridorVerifier
from verifiers.controlled_descent import ControlledDescentVerifier
from verifiers.pitch_trimming import PitchTrimmingVerifier

# State-based verifiers
from verifiers.state_based import (
    SafeVelocityVerifier,
    StableRotationVerifier,
    OnTargetVerifier,
    SafeLandingConditionVerifier,
    DescendingVerifier,
    InCorridorVerifier,
)


def load_checkpoint(checkpoint_path: str, device: str = "mps"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['cfg']
    
    policy: TEDiUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
    
    if 'state_dicts' in checkpoint and 'ema_model' in checkpoint['state_dicts']:
        policy.load_state_dict(checkpoint['state_dicts']['ema_model'])
    else:
        policy.load_state_dict(checkpoint['state_dicts']['model'])
    
    policy.to(device)
    policy.eval()
    
    # Load normalizer
    if 'normalizer' in checkpoint:
        policy.normalizer.load_state_dict(checkpoint['normalizer'])
    
    return policy, cfg


def sample_k_candidates_batched(policy, obs_dict, k, device):
    """Sample K action candidates in a single batched forward pass."""
    obs_expanded = obs_dict['obs'].expand(k, -1, -1)
    policy.action_buffer = None  # Reset buffer for independent samples
    
    with torch.no_grad():
        action_dict = policy.predict_action({'obs': obs_expanded})
    
    actions = action_dict['action'].cpu().numpy()
    return [np.argmax(actions[i], axis=1) for i in range(k)]


class HybridVerifierScorer:
    """
    Scorer that uses both image-based and state-based verifiers.
    
    Image verifiers: Check visual properties (position, flames, etc.)
    State verifiers: Check dynamic properties (velocity, trajectory, etc.)
    """
    
    def __init__(self):
        # Image-based verifiers
        self.image_verifiers = {
            'vertical_corridor': VerticalCorridorVerifier(),
            'controlled_descent': ControlledDescentVerifier(),
            'pitch_trimming': PitchTrimmingVerifier(),
        }
        
        # State-based verifiers
        self.state_verifiers = {
            'safe_velocity': SafeVelocityVerifier(),
            'stable_rotation': StableRotationVerifier(),
            'on_target': OnTargetVerifier(),
            'safe_landing': SafeLandingConditionVerifier(),
            'descending': DescendingVerifier(),
            'in_corridor': InCorridorVerifier(),
        }
        
        # Weights for each verifier
        self.weights = {
            # Image-based
            'vertical_corridor': 2.0,  # Visual confirmation of position
            'controlled_descent': 1.0,
            'pitch_trimming': 1.0,
            # State-based (these are more accurate)
            'safe_velocity': 3.0,      # Critical for landing
            'stable_rotation': 2.0,    # Important for control
            'on_target': 3.0,          # Trajectory prediction is key
            'safe_landing': 2.0,       # Final approach
            'descending': 1.0,         # Basic sanity check
            'in_corridor': 2.0,        # More accurate than image version
        }
    
    def score(self, frame_bgr: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict]:
        """
        Score a frame+observation pair using all verifiers.
        
        Returns:
            (total_score, detailed_results)
        """
        results = {}
        total_score = 0.0
        
        # Run image-based verifiers
        for name, verifier in self.image_verifiers.items():
            passed = verifier(frame_bgr)
            results[f'img_{name}'] = bool(passed)
            if passed:
                total_score += self.weights.get(name, 1.0)
        
        # Run state-based verifiers
        for name, verifier in self.state_verifiers.items():
            passed = verifier(obs)
            results[f'state_{name}'] = bool(passed)
            if passed:
                total_score += self.weights.get(name, 1.0)
        
        results['total_score'] = total_score
        return total_score, results


def create_synced_env(seed: int, action_history: list):
    """Create environment synced to current state by replaying action history."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    
    for action in action_history:
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break
    
    return env, obs


def evaluate_candidate_parallel(
    seed: int,
    action_history: list,
    candidate_actions: np.ndarray,
    scorer: HybridVerifierScorer,
    n_lookahead: int
) -> Tuple[float, List[Dict]]:
    """Evaluate a candidate by simulating in a parallel environment."""
    env, obs = create_synced_env(seed, action_history)
    
    total_score = 0.0
    frame_results = []
    
    for i, action in enumerate(candidate_actions[:n_lookahead]):
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        frame_rgb = env.render()
        if frame_rgb is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            score, results = scorer.score(frame_bgr, obs)
            total_score += score
            results['step'] = i
            results['reward'] = reward
            frame_results.append(results)
        
        if terminated or truncated:
            # Penalty for crashing
            if terminated and reward < 0:
                total_score -= 5.0
            break
    
    env.close()
    return total_score, frame_results


def run_evaluation(
    policy,
    cfg,
    k: int,
    n_episodes: int,
    n_lookahead: int,
    device: str,
    seed: int,
    use_verifiers: bool = True,
):
    """Run evaluation with optional verifier-based selection."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    n_obs_steps = cfg.n_obs_steps
    
    scorer = HybridVerifierScorer()
    
    episode_results = []
    episode_rewards = []
    successes = []
    
    method_name = "Verifier" if use_verifiers else "Random"
    pbar = tqdm(range(n_episodes), desc=f"{method_name} Eval", unit="ep")
    
    for ep in pbar:
        episode_seed = seed + ep
        obs, _ = env.reset(seed=episode_seed)
        
        # Initialize observation history (same as working script)
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()  # Reset at start of episode
        
        action_history = []
        action_queue = []  # Use action queue like working script
        
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 1000:
            if len(action_queue) > 0:
                # Still executing previous action sequence
                action = action_queue.pop(0)
            else:
                # Need new actions - decision point
                # Build observation tensor (NO explicit normalization - matches working script)
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_dict = {'obs': obs_tensor}
                
                # Sample K candidates
                candidates = sample_k_candidates_batched(policy, obs_dict, k, device)
                
                if use_verifiers and len(action_history) > 0:
                    # Evaluate each candidate with verifiers
                    best_idx = 0
                    best_score = -float('inf')
                    
                    for i, candidate in enumerate(candidates):
                        score, _ = evaluate_candidate_parallel(
                            seed=episode_seed,
                            action_history=action_history,
                            candidate_actions=candidate,
                            scorer=scorer,
                            n_lookahead=n_lookahead
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    
                    selected_actions = candidates[best_idx]
                else:
                    # Random selection
                    selected_actions = candidates[random.randint(0, k-1)]
                
                # Queue actions (already discrete from sample_k_candidates_batched)
                action_queue = list(selected_actions)
                action = action_queue.pop(0)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            obs_history.append(obs)
            action_history.append(int(action))
            total_reward += reward
            step += 1
        
        success = total_reward >= 200
        successes.append(success)
        episode_rewards.append(total_reward)
        
        episode_results.append({
            'episode': ep,
            'reward': total_reward,
            'success': success,
            'steps': step,
        })
        
        # Update progress bar with running statistics
        current_mean = np.mean(episode_rewards)
        current_success = np.mean(successes) * 100
        pbar.set_postfix({
            'mean': f'{current_mean:.1f}',
            'success': f'{current_success:.0f}%',
            'last': f'{total_reward:.1f}'
        })
    
    pbar.close()
    env.close()
    return episode_results


def print_results(results: List[Dict], method: str):
    """Print summary of results."""
    rewards = [r['reward'] for r in results]
    successes = [r['success'] for r in results]
    
    print(f"\n{'='*50}")
    print(f"Results: {method}")
    print(f"{'='*50}")
    print(f"Episodes:     {len(results)}")
    print(f"Success Rate: {sum(successes)/len(successes)*100:.1f}%")
    print(f"Mean Reward:  {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Min Reward:   {np.min(rewards):.1f}")
    print(f"Max Reward:   {np.max(rewards):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k', type=int, default=5, help='Number of candidates')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--lookahead', type=int, default=8, help='Steps to simulate ahead')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--compare', action='store_true', help='Compare verifiers vs random')
    
    args = parser.parse_args()
    
    print("Loading checkpoint...")
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    if args.temperature != 1.0:
        policy.set_temperature(args.temperature)
        print(f"Temperature set to {args.temperature}")
    
    if args.compare:
        # Run both methods for comparison
        print("\n" + "="*60)
        print("RANDOM SELECTION (baseline)")
        print("="*60)
        random_results = run_evaluation(
            policy, cfg, args.k, args.n_episodes, args.lookahead,
            args.device, args.seed, use_verifiers=False
        )
        print_results(random_results, "Random Selection")
        
        print("\n" + "="*60)
        print("HYBRID VERIFIER SELECTION (image + state)")
        print("="*60)
        verifier_results = run_evaluation(
            policy, cfg, args.k, args.n_episodes, args.lookahead,
            args.device, args.seed, use_verifiers=True
        )
        print_results(verifier_results, "Hybrid Verifier Selection")
        
        # Comparison
        random_success = sum(r['success'] for r in random_results) / len(random_results)
        verifier_success = sum(r['success'] for r in verifier_results) / len(verifier_results)
        
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Random Success:   {random_success*100:.1f}%")
        print(f"Verifier Success: {verifier_success*100:.1f}%")
        print(f"Improvement:      {(verifier_success - random_success)*100:+.1f}%")
        
    else:
        # Just run verifier selection
        results = run_evaluation(
            policy, cfg, args.k, args.n_episodes, args.lookahead,
            args.device, args.seed, use_verifiers=True
        )
        print_results(results, "Hybrid Verifier Selection")


if __name__ == "__main__":
    main()

