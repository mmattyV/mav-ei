#!/usr/bin/env python3
"""
SDP Evaluation with Screenshot Verifiers for MAV-EI.

This script:
1. Samples K candidates from SDP and picks one at random
2. Captures screenshots after EVERY action step
3. Runs Ryan's 4 image-based verifiers on each screenshot
4. Reports verifier pass rates and correlations with success

Image Verifiers (from Ryan):
- VerticalCorridorVerifier: Is lander within the landing corridor?
- ControlledDescentVerifier: Is lander using main engine for braking?
- PitchTrimmingVerifier: Is lander correcting its tilt properly?
- FinalSuccessVerifier: Has lander successfully landed?
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
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy

# Import verifiers
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
    obs_dict_batched = {'obs': obs_expanded}
    policy.action_buffer = None
    
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict_batched)
    
    actions = action_dict['action'].cpu().numpy()
    candidates = [np.argmax(actions[i], axis=1) for i in range(k)]
    return candidates


def capture_frame(env) -> np.ndarray:
    """
    Capture current frame from environment.
    Returns BGR image for cv2 compatibility with verifiers.
    """
    frame_rgb = env.render()
    if frame_rgb is None:
        return None
    # Convert RGB to BGR for cv2/verifier compatibility
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


class ScreenshotVerifierMonitor:
    """
    Monitors episode quality using Ryan's image-based verifiers.
    Captures screenshots after each action and runs all verifiers.
    """
    
    def __init__(self, save_screenshots: bool = False, screenshot_dir: str = None):
        # Initialize verifiers
        self.verifiers = {
            'vertical_corridor': VerticalCorridorVerifier(),
            'controlled_descent': ControlledDescentVerifier(),
            'pitch_trimming': PitchTrimmingVerifier(),
            'final_success': FinalSuccessVerifier(),
        }
        
        self.save_screenshots = save_screenshots
        self.screenshot_dir = screenshot_dir
        
        if save_screenshots and screenshot_dir:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_frame(self, frame: np.ndarray) -> dict:
        """Run all verifiers on a single frame."""
        results = {}
        n_approvals = 0
        
        for name, verifier in self.verifiers.items():
            approved = verifier(frame)
            results[name] = approved
            if approved:
                n_approvals += 1
        
        results['n_approvals'] = n_approvals
        results['approval_rate'] = n_approvals / len(self.verifiers)
        
        return results
    
    def save_frame(self, frame: np.ndarray, episode: int, step: int, suffix: str = ""):
        """Save a screenshot to disk."""
        if self.save_screenshots and self.screenshot_dir:
            filename = f"ep{episode:03d}_step{step:04d}{suffix}.png"
            filepath = Path(self.screenshot_dir) / filename
            cv2.imwrite(str(filepath), frame)


def evaluate_sdp_with_verifiers(
    policy: TEDiUnetLowdimPolicy,
    cfg,
    monitor: ScreenshotVerifierMonitor,
    k: int = 5,
    n_episodes: int = 20,
    max_steps: int = 1000,
    device: str = "mps",
    seed: int = 42,
):
    """
    Evaluate SDP with K candidates (random selection) and screenshot verifier monitoring.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Use rgb_array mode to capture frames
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    
    print(f"\n{'='*60}")
    print("SDP Evaluation with Screenshot Verifiers")
    print(f"{'='*60}")
    print(f"  K (candidates): {k}")
    print(f"  Selection: RANDOM")
    print(f"  Verifiers: {list(monitor.verifiers.keys())}")
    print(f"  n_episodes: {n_episodes}")
    print()
    
    # Results storage
    episode_rewards = []
    episode_lengths = []
    successes = []
    all_episode_stats = []
    
    pbar = tqdm(range(n_episodes), desc="Evaluating", unit="episode")
    
    for ep in pbar:
        obs, info = env.reset(seed=seed + ep)
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        total_reward = 0.0
        steps = 0
        done = False
        action_queue = []
        
        # Per-episode verifier tracking
        episode_verifier_results = []
        
        while not done and steps < max_steps:
            # Capture frame and run verifiers AFTER each action
            frame = capture_frame(env)
            if frame is not None:
                verifier_result = monitor.evaluate_frame(frame)
                verifier_result['step'] = steps
                episode_verifier_results.append(verifier_result)
                monitor.save_frame(frame, ep, steps)
            
            # Get next action
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                # Stack observations for SDP
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_dict = {'obs': obs_tensor}
                
                # Sample K candidates and pick one at random
                candidates = sample_k_candidates_batched(policy, obs_dict, k, device)
                selected_idx = random.randint(0, k - 1)
                selected_actions = candidates[selected_idx]
                
                action_queue = list(selected_actions)
                action = action_queue.pop(0)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            obs_history.append(next_obs)
            total_reward += reward
            steps += 1
        
        # Capture and verify FINAL frame
        final_frame = capture_frame(env)
        if final_frame is not None:
            final_result = monitor.evaluate_frame(final_frame)
            final_result['step'] = steps
            final_result['is_final'] = True
            episode_verifier_results.append(final_result)
            monitor.save_frame(final_frame, ep, steps, "_final")
        
        # Record episode results
        success = total_reward >= 200
        successes.append(success)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Aggregate verifier stats for this episode
        ep_stats = {
            'episode': ep,
            'reward': total_reward,
            'success': success,
            'steps': steps,
            'n_frames': len(episode_verifier_results),
        }
        
        # Calculate pass rates for each verifier
        for vname in monitor.verifiers.keys():
            passes = sum(1 for r in episode_verifier_results if r.get(vname, False))
            ep_stats[f'{vname}_pass_rate'] = passes / len(episode_verifier_results) if episode_verifier_results else 0
            ep_stats[f'{vname}_passes'] = passes
        
        # Check final frame specifically
        if episode_verifier_results and episode_verifier_results[-1].get('is_final'):
            for vname in monitor.verifiers.keys():
                ep_stats[f'{vname}_final'] = episode_verifier_results[-1].get(vname, False)
        
        ep_stats['mean_approval_rate'] = np.mean([r['approval_rate'] for r in episode_verifier_results]) if episode_verifier_results else 0
        
        all_episode_stats.append(ep_stats)
        
        # Update progress bar
        current_mean = np.mean(episode_rewards)
        current_success = np.mean(successes) * 100
        pbar.set_postfix({
            'mean': f'{current_mean:.1f}',
            'success': f'{current_success:.0f}%',
            'verifiers': f'{ep_stats["mean_approval_rate"]*100:.0f}%'
        })
    
    pbar.close()
    env.close()
    
    # Compile results
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
        'episode_stats': all_episode_stats,
    }
    
    # Aggregate verifier statistics across all episodes
    for vname in monitor.verifiers.keys():
        key = f'{vname}_pass_rate'
        results[f'mean_{key}'] = np.mean([s[key] for s in all_episode_stats])
    
    results['mean_overall_approval'] = np.mean([s['mean_approval_rate'] for s in all_episode_stats])
    
    return results


def print_results(results):
    """Print comprehensive results."""
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n[Task Performance]")
    print(f"  K (candidates):   {results['k']}")
    print(f"  Selection:        {results['selection_method'].upper()}")
    print(f"  Mean Reward:      {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min/Max:          {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"  Mean Length:      {results['mean_length']:.1f} steps")
    print(f"  Success Rate:     {results['success_rate']*100:.1f}% ({results['n_successes']}/{results['n_episodes']})")
    
    print("\n[Image Verifier Pass Rates (averaged across all frames)]")
    print(f"  Vertical Corridor:   {results['mean_vertical_corridor_pass_rate']*100:.1f}%")
    print(f"  Controlled Descent:  {results['mean_controlled_descent_pass_rate']*100:.1f}%")
    print(f"  Pitch Trimming:      {results['mean_pitch_trimming_pass_rate']*100:.1f}%")
    print(f"  Final Success:       {results['mean_final_success_pass_rate']*100:.1f}%")
    print(f"  Overall Approval:    {results['mean_overall_approval']*100:.1f}%")
    
    # Analyze correlation between verifiers and success
    print("\n[Verifier vs Success Correlation]")
    successful_eps = [s for s in results['episode_stats'] if s['success']]
    failed_eps = [s for s in results['episode_stats'] if not s['success']]
    
    if successful_eps and failed_eps:
        for vname in ['vertical_corridor', 'controlled_descent', 'pitch_trimming', 'final_success']:
            key = f'{vname}_pass_rate'
            success_rate = np.mean([s[key] for s in successful_eps])
            fail_rate = np.mean([s[key] for s in failed_eps])
            diff = success_rate - fail_rate
            print(f"  {vname}: +{diff*100:.1f}% in successful episodes")
    
    print("\n[Per-Episode Details (first 10)]")
    print("-" * 85)
    print(f"{'Ep':>3} {'Reward':>8} {'OK':>4} {'Steps':>5} {'VC':>6} {'CD':>6} {'PT':>6} {'FS':>6} {'Avg':>6}")
    print("-" * 85)
    
    for s in results['episode_stats'][:10]:
        ok = "✓" if s['success'] else "✗"
        print(f"{s['episode']:>3} {s['reward']:>8.1f} {ok:>4} {s['steps']:>5} "
              f"{s['vertical_corridor_pass_rate']*100:>5.0f}% "
              f"{s['controlled_descent_pass_rate']*100:>5.0f}% "
              f"{s['pitch_trimming_pass_rate']*100:>5.0f}% "
              f"{s['final_success_pass_rate']*100:>5.0f}% "
              f"{s['mean_approval_rate']*100:>5.0f}%")
    
    if len(results['episode_stats']) > 10:
        print(f"  ... ({len(results['episode_stats']) - 10} more episodes)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SDP with Screenshot Verifiers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with K=5
  python evaluate_sdp_with_verifiers.py --checkpoint path/to/checkpoint.ckpt --k 5
  
  # Save screenshots for analysis
  python evaluate_sdp_with_verifiers.py --checkpoint path/to/checkpoint.ckpt --k 5 --save_screenshots
  
  # Quick test
  python evaluate_sdp_with_verifiers.py --checkpoint path/to/checkpoint.ckpt --k 5 --n_episodes 5
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_screenshots', action='store_true',
                        help='Save screenshots to disk')
    parser.add_argument('--screenshot_dir', type=str, default=None,
                        help='Directory to save screenshots (default: screenshots/eval_<timestamp>)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MAV-EI: SDP Evaluation with Screenshot Verifiers")
    print("=" * 60)
    print()
    
    # Setup screenshot directory
    screenshot_dir = args.screenshot_dir
    if args.save_screenshots and screenshot_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = project_root / "screenshots" / f"eval_{timestamp}"
    
    # Load SDP policy
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Create verifier monitor
    monitor = ScreenshotVerifierMonitor(
        save_screenshots=args.save_screenshots,
        screenshot_dir=str(screenshot_dir) if screenshot_dir else None,
    )
    
    print(f"Loaded verifiers: {list(monitor.verifiers.keys())}")
    
    # Run evaluation
    results = evaluate_sdp_with_verifiers(
        policy=policy,
        cfg=cfg,
        monitor=monitor,
        k=args.k,
        n_episodes=args.n_episodes,
        device=args.device,
        seed=args.seed,
    )
    
    print_results(results)
    
    if args.save_screenshots:
        print(f"Screenshots saved to: {screenshot_dir}")


if __name__ == "__main__":
    main()

