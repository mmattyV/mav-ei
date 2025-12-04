#!/usr/bin/env python3
"""
MAV-EI: Debug Verifier Selection

This script runs verifier-based selection and saves:
- Screenshots at each step of the REAL trajectory
- Verifier scores for each screenshot
- Organized by success/failure

Output structure:
  debug_screenshots/
    successful/
      ep_00_reward_245.3/
        step_000.png
        step_001.png
        ...
        scores.json
    failed/
      ep_01_reward_-50.2/
        step_000.png
        ...
        scores.json
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


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    else:
        return obj

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy

from verifiers.vertical_corridor import VerticalCorridorVerifier
from verifiers.controlled_descent import ControlledDescentVerifier
from verifiers.pitch_trimming import PitchTrimmingVerifier
# from verifiers.final_success import FinalSuccessVerifier  # Disabled - gives false positives


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
    
    return policy, cfg


def sample_k_candidates_batched(policy, obs_dict, k, device):
    obs_expanded = obs_dict['obs'].expand(k, -1, -1)
    policy.action_buffer = None
    
    with torch.no_grad():
        action_dict = policy.predict_action({'obs': obs_expanded})
    
    actions = action_dict['action'].cpu().numpy()
    return [np.argmax(actions[i], axis=1) for i in range(k)]


class VerifierScorer:
    def __init__(self):
        self.verifiers = {
            'vertical_corridor': VerticalCorridorVerifier(),
            'controlled_descent': ControlledDescentVerifier(),
            'pitch_trimming': PitchTrimmingVerifier(),
            # 'final_success': FinalSuccessVerifier(),  # Disabled - gives false positives
        }
    
    def score_frame_detailed(self, frame_bgr):
        """Score a frame and return detailed results.
        
        Weighting: vertical_corridor counts 2x (since it's the key differentiator)
        """
        results = {}
        total_score = 0
        
        # Weights for each verifier (vertical_corridor is most important)
        weights = {
            'vertical_corridor': 2,  # 2x weight - key differentiator
            'controlled_descent': 1,
            'pitch_trimming': 1,
            # 'final_success': 1,  # Disabled - gives false positives
        }
        
        for name, verifier in self.verifiers.items():
            passed = verifier(frame_bgr)
            results[name] = passed
            if passed:
                total_score += weights.get(name, 1)
        
        results['total_score'] = total_score
        return results


def create_synced_env(seed: int, action_history: list):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    
    for action in action_history:
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break
    
    return env, obs


def evaluate_candidate_parallel(seed, action_history, candidate_actions, scorer, n_lookahead):
    env, obs = create_synced_env(seed, action_history)
    
    total_score = 0
    achieved_final_success = False  # Always False now (final_success verifier disabled)
    
    for action in candidate_actions[:n_lookahead]:
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        frame_rgb = env.render()
        if frame_rgb is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            results = scorer.score_frame_detailed(frame_bgr)
            total_score += results['total_score']
        
        if terminated or truncated:
            break
    
    env.close()
    return total_score, achieved_final_success


def run_debug_evaluation(policy, cfg, k, n_episodes, n_lookahead, device, seed, output_dir):
    """Run verifier-based selection with full debugging output."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    n_obs_steps = cfg.n_obs_steps
    
    scorer = VerifierScorer()
    
    # Create output directories
    success_dir = output_dir / "successful"
    failed_dir = output_dir / "failed"
    success_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    episode_results = []
    
    pbar = tqdm(range(n_episodes), desc="Debug Eval", unit="ep")
    
    for ep in pbar:
        episode_seed = seed + ep
        obs, info = env.reset(seed=episode_seed)
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        action_history = []
        
        # Track this episode's data
        episode_frames = []
        episode_scores = []
        episode_actions = []
        episode_rewards = []
        selection_data = []  # Data about each selection decision
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        while not done and steps < 1000:
            # Capture frame BEFORE action
            frame_rgb = env.render()
            if frame_rgb is not None:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                episode_frames.append(frame_bgr)
                
                # Score current frame
                frame_results = scorer.score_frame_detailed(frame_bgr)
                frame_results['step'] = steps
                episode_scores.append(frame_results)
            
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                # Decision point - record selection data
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                
                candidates = sample_k_candidates_batched(policy, {'obs': obs_tensor}, k, device)
                
                # Evaluate each candidate
                candidate_scores = []
                
                for i, candidate in enumerate(candidates):
                    score, achieved_final = evaluate_candidate_parallel(
                        seed=episode_seed,
                        action_history=action_history,
                        candidate_actions=candidate,
                        scorer=scorer,
                        n_lookahead=n_lookahead,
                    )
                    candidate_scores.append({
                        'candidate_idx': i,
                        'actions': candidate.tolist(),
                        'score': score,
                        'achieved_final_success': achieved_final,
                    })
                
                # Pick highest score (no early exit - FinalSuccess can give false positives)
                selected_idx = np.argmax([c['score'] for c in candidate_scores])
                
                # Record selection decision
                selection_data.append({
                    'step': steps,
                    'candidates': candidate_scores,
                    'selected_idx': selected_idx,
                    'selected_score': candidate_scores[selected_idx]['score'] if selected_idx < len(candidate_scores) else 0,
                    'all_scores': [c['score'] for c in candidate_scores[:selected_idx+1]],
                })
                
                action_queue = list(candidates[selected_idx])
                action = action_queue.pop(0)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            action_history.append(int(action))
            episode_actions.append(int(action))
            episode_rewards.append(float(reward))
            
            done = terminated or truncated
            obs_history.append(next_obs)
            total_reward += reward
            steps += 1
        
        # Capture final frame
        final_frame_rgb = env.render()
        if final_frame_rgb is not None:
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)
            episode_frames.append(final_frame_bgr)
            final_results = scorer.score_frame_detailed(final_frame_bgr)
            final_results['step'] = steps
            final_results['is_final'] = True
            episode_scores.append(final_results)
        
        # Determine success
        success = total_reward >= 200
        
        # Save episode data
        if success:
            ep_dir = success_dir / f"ep_{ep:02d}_reward_{total_reward:.1f}"
        else:
            ep_dir = failed_dir / f"ep_{ep:02d}_reward_{total_reward:.1f}"
        
        ep_dir.mkdir(parents=True, exist_ok=True)
        
        # Save screenshots
        for i, frame in enumerate(episode_frames):
            cv2.imwrite(str(ep_dir / f"step_{i:03d}.png"), frame)
        
        # Save scores and metadata
        episode_metadata = {
            'episode': ep,
            'seed': episode_seed,
            'total_reward': total_reward,
            'success': success,
            'n_steps': steps,
            'n_frames': len(episode_frames),
            'frame_scores': episode_scores,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'selection_decisions': selection_data,
            'verifier_summary': {
                'vertical_corridor_pass_rate': sum(1 for s in episode_scores if s.get('vertical_corridor')) / len(episode_scores) if episode_scores else 0,
                'controlled_descent_pass_rate': sum(1 for s in episode_scores if s.get('controlled_descent')) / len(episode_scores) if episode_scores else 0,
                'pitch_trimming_pass_rate': sum(1 for s in episode_scores if s.get('pitch_trimming')) / len(episode_scores) if episode_scores else 0,
                'final_success_count': sum(1 for s in episode_scores if s.get('final_success')),
                'mean_total_score': np.mean([s['total_score'] for s in episode_scores]) if episode_scores else 0,
            }
        }
        
        with open(ep_dir / "scores.json", 'w') as f:
            json.dump(convert_to_serializable(episode_metadata), f, indent=2)
        
        episode_results.append({
            'episode': ep,
            'reward': total_reward,
            'success': success,
            'n_steps': steps,
        })
        
        pbar.set_postfix({
            'reward': f'{total_reward:.0f}',
            'success': '✓' if success else '✗',
        })
    
    pbar.close()
    env.close()
    
    # Save overall summary
    n_successes = sum(1 for r in episode_results if r['success'])
    summary = {
        'n_episodes': n_episodes,
        'k': k,
        'lookahead': n_lookahead,
        'seed': seed,
        'n_successes': n_successes,
        'success_rate': n_successes / n_episodes,
        'mean_reward': np.mean([r['reward'] for r in episode_results]),
        'std_reward': np.std([r['reward'] for r in episode_results]),
        'episode_results': episode_results,
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Debug Verifier Selection')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--lookahead', type=int, default=4)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (>1.0 = more diverse candidates)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: debug_screenshots/<timestamp>)')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "debug_screenshots" / f"debug_{timestamp}"
    
    print("=" * 70)
    print("MAV-EI: Debug Verifier Selection")
    print("=" * 70)
    print()
    print(f"  K (candidates):      {args.k}")
    print(f"  Lookahead steps:     {args.lookahead}")
    print(f"  Episodes:            {args.n_episodes}")
    print(f"  Temperature:         {args.temperature}")
    print(f"  Output directory:    {output_dir}")
    print()
    
    # Load policy
    print("Loading SDP policy...")
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Set temperature for sampling diversity
    if args.temperature != 1.0:
        policy.set_temperature(args.temperature)
        print(f"  Temperature set to {args.temperature} for more diverse sampling")
    print()
    
    # Run debug evaluation
    print("Running debug evaluation...")
    print("(Saving screenshots and verifier scores for each step)")
    print()
    
    summary = run_debug_evaluation(
        policy, cfg, args.k, args.n_episodes, args.lookahead,
        args.device, args.seed, output_dir
    )
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Success Rate:  {summary['success_rate']*100:.1f}% ({summary['n_successes']}/{summary['n_episodes']})")
    print(f"  Mean Reward:   {summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
    print()
    print(f"  Output saved to: {output_dir}")
    print()
    print("  Directory structure:")
    print("    successful/")
    print("      ep_XX_reward_YYY/")
    print("        step_000.png, step_001.png, ...")
    print("        scores.json  (verifier scores + selection decisions)")
    print("    failed/")
    print("      ep_XX_reward_YYY/")
    print("        ...")
    print("    summary.json")
    print()
    print("  Examine scores.json for:")
    print("    - frame_scores: verifier results at each step")
    print("    - selection_decisions: which candidates were considered and why")
    print("    - verifier_summary: pass rates for each verifier")


if __name__ == "__main__":
    main()

