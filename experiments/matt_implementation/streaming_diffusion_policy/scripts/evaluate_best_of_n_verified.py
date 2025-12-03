#!/usr/bin/env python3
"""
Best-of-N with Verifier-Based Selection for MAV-EI.

Uses PARALLEL ENVIRONMENTS to properly evaluate each candidate:
1. Sample K action candidates from SDP
2. For EACH candidate:
   - Create fresh environment
   - Replay action history to sync to current state
   - Simulate candidate's actions
   - Take screenshots and run image verifiers
   - Sum verifier scores
3. SELECT candidate with HIGHEST verifier score
4. Execute that candidate in the real environment
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
from verifiers.final_success import FinalSuccessVerifier


def load_checkpoint(checkpoint_path: str, device: str = "mps"):
    """Load a trained SDP checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['cfg']
    
    policy: TEDiUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
    
    if 'state_dicts' in checkpoint and 'ema_model' in checkpoint['state_dicts']:
        print("Using EMA weights")
        policy.load_state_dict(checkpoint['state_dicts']['ema_model'])
    else:
        print("Using regular weights")
        policy.load_state_dict(checkpoint['state_dicts']['model'])
    
    policy.to(device)
    policy.eval()
    
    return policy, cfg


def sample_k_candidates_batched(policy, obs_dict, k, device):
    """Sample K action candidates from SDP using batched inference."""
    obs_expanded = obs_dict['obs'].expand(k, -1, -1)
    policy.action_buffer = None
    
    with torch.no_grad():
        action_dict = policy.predict_action({'obs': obs_expanded})
    
    actions = action_dict['action'].cpu().numpy()
    return [np.argmax(actions[i], axis=1) for i in range(k)]


class VerifierScorer:
    """Score frames using image verifiers."""
    
    def __init__(self):
        self.verifiers = {
            'vertical_corridor': VerticalCorridorVerifier(),
            'controlled_descent': ControlledDescentVerifier(),
            'pitch_trimming': PitchTrimmingVerifier(),
        }
        # Final success verifier is special - if True, we pick this candidate immediately
        self.final_success_verifier = FinalSuccessVerifier()
    
    def score_frame(self, frame_bgr):
        """
        Score a single BGR frame.
        
        Returns:
            (score, is_final_success): score from verifiers, and whether final success was achieved
        """
        score = 0
        for verifier in self.verifiers.values():
            if verifier(frame_bgr):
                score += 1
        
        # Check final success separately
        is_final_success = self.final_success_verifier(frame_bgr)
        if is_final_success:
            score += 100  # Huge bonus for achieving final success
        
        return score, is_final_success


def create_synced_env(seed: int, action_history: list):
    """Create fresh environment and replay action history to sync it."""
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
    candidate_actions: list,
    scorer: VerifierScorer,
    n_lookahead: int,
):
    """
    Evaluate a candidate using a parallel environment.
    
    Returns:
        (score, achieved_final_success): Total verifier score and whether final success was achieved
    """
    env, obs = create_synced_env(seed, action_history)
    
    total_score = 0
    achieved_final_success = False
    
    for action in candidate_actions[:n_lookahead]:
        obs, reward, terminated, truncated, info = env.step(int(action))
        
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


def evaluate_best_of_n_verified(
    policy: TEDiUnetLowdimPolicy,
    cfg,
    k: int = 5,
    n_episodes: int = 20,
    max_steps: int = 1000,
    n_lookahead: int = 4,
    device: str = "mps",
    seed: int = 42,
):
    """
    Evaluate Best-of-N with VERIFIER-BASED SELECTION using parallel environments.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    n_obs_steps = cfg.n_obs_steps
    
    scorer = VerifierScorer()
    
    print(f"\n{'='*60}")
    print("Best-of-N with VERIFIER-BASED SELECTION")
    print(f"{'='*60}")
    print(f"  K (candidates): {k}")
    print(f"  Selection: VERIFIER SCORE (highest)")
    print(f"  Lookahead steps: {n_lookahead}")
    print(f"  Verifiers: {list(scorer.verifiers.keys())}")
    print(f"  n_episodes: {n_episodes}")
    print()
    print("  Strategy: Parallel environments (create, replay, simulate, score)")
    print()
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    total_selections = 0
    
    pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode")
    
    for ep in pbar:
        episode_seed = seed + ep
        obs, info = env.reset(seed=episode_seed)
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        action_history = []
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        while not done and steps < max_steps:
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                total_selections += 1
                
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_dict = {'obs': obs_tensor}
                
                candidates = sample_k_candidates_batched(policy, obs_dict, k, device)
                
                # Evaluate each candidate in parallel environment
                scores = []
                best_idx = None
                
                for i, candidate in enumerate(candidates):
                    score, achieved_final_success = evaluate_candidate_parallel(
                        seed=episode_seed,
                        action_history=action_history,
                        candidate_actions=candidate,
                        scorer=scorer,
                        n_lookahead=n_lookahead,
                    )
                    scores.append(score)
                    
                    # OPTIMIZATION: If this candidate achieves final success, pick it immediately!
                    if achieved_final_success:
                        best_idx = i
                        break
                
                # If no candidate achieved final success, pick highest score
                if best_idx is None:
                    best_idx = np.argmax(scores)
                
                selected_actions = candidates[best_idx]
                
                action_queue = list(selected_actions)
                action = action_queue.pop(0)
            
            # Execute in real environment
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            action_history.append(action)
            
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
            'mean': f'{current_mean:.1f}',
            'success': f'{current_success:.0f}%',
            'last': f'{total_reward:.1f}'
        })
    
    pbar.close()
    env.close()
    
    results = {
        'k': k,
        'selection_method': 'verifier_score',
        'n_steps_lookahead': n_lookahead,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes),
        'n_successes': sum(successes),
        'n_episodes': n_episodes,
        'total_selections': total_selections,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Best-of-N with Verifier-Based Selection (Parallel Environments)',
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--lookahead', type=int, default=4)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAV-EI: Best-of-N with VERIFIER-BASED Selection")
    print("=" * 60)
    print()
    
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    results = evaluate_best_of_n_verified(
        policy=policy,
        cfg=cfg,
        k=args.k,
        n_episodes=args.n_episodes,
        n_lookahead=args.lookahead,
        device=args.device,
        seed=args.seed,
    )
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  K (candidates):        {results['k']}")
    print(f"  Selection:             {results['selection_method'].upper()}")
    print(f"  Lookahead steps:       {results['n_steps_lookahead']}")
    print(f"  Mean Reward:           {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min/Max:               {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean Length:           {results['mean_length']:.1f} steps")
    print(f"  Success Rate:          {results['success_rate']*100:.1f}% ({results['n_successes']}/{results['n_episodes']})")
    print(f"  Total Selections:      {results['total_selections']}")
    print()
    print("Run compare_selection_methods.py to compare with random baseline!")


if __name__ == "__main__":
    main()
