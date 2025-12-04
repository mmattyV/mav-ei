#!/usr/bin/env python3
"""
MAV-EI Ablation Study for Paper

Runs all experiments needed for "Excellent" on Results rubric:
1. Random baseline (no verifiers)
2. Image-only verifiers
3. State-only verifiers  
4. Hybrid verifiers (equal weights)
5. Hybrid verifiers (tuned weights)

Outputs:
- results/ablation_results.json - Raw data for tables
- results/success_rate_comparison.png - Bar chart
- results/reward_distribution.png - Box plot
- results/verifier_analysis.png - Pass rates by success/failure
- results/latex_tables.txt - Copy-paste LaTeX code
"""

import os
import sys
import argparse
import random
import json
import time
import numpy as np
import torch
import gymnasium as gym
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving

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


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    use_verifiers: bool
    use_image_verifiers: bool
    use_state_verifiers: bool
    use_tuned_weights: bool
    description: str


# Define all experiments
EXPERIMENTS = [
    ExperimentConfig(
        name="random_baseline",
        use_verifiers=False,
        use_image_verifiers=False,
        use_state_verifiers=False,
        use_tuned_weights=False,
        description="Random selection (no verifiers)"
    ),
    ExperimentConfig(
        name="image_only",
        use_verifiers=True,
        use_image_verifiers=True,
        use_state_verifiers=False,
        use_tuned_weights=False,
        description="Image-based verifiers only"
    ),
    ExperimentConfig(
        name="state_only",
        use_verifiers=True,
        use_image_verifiers=False,
        use_state_verifiers=True,
        use_tuned_weights=False,
        description="State-based verifiers only"
    ),
    ExperimentConfig(
        name="hybrid_equal",
        use_verifiers=True,
        use_image_verifiers=True,
        use_state_verifiers=True,
        use_tuned_weights=False,
        description="Hybrid verifiers (equal weights)"
    ),
    ExperimentConfig(
        name="hybrid_tuned",
        use_verifiers=True,
        use_image_verifiers=True,
        use_state_verifiers=True,
        use_tuned_weights=True,
        description="Hybrid verifiers (tuned weights)"
    ),
]


class ConfigurableVerifierScorer:
    """Scorer with configurable verifier sets and weights."""
    
    def __init__(
        self,
        use_image: bool = True,
        use_state: bool = True,
        use_tuned_weights: bool = False
    ):
        self.use_image = use_image
        self.use_state = use_state
        
        # Image-based verifiers
        self.image_verifiers = {
            'img_vertical_corridor': VerticalCorridorVerifier(),
            'img_controlled_descent': ControlledDescentVerifier(),
            'img_pitch_trimming': PitchTrimmingVerifier(),
        } if use_image else {}
        
        # State-based verifiers
        self.state_verifiers = {
            'state_safe_velocity': SafeVelocityVerifier(),
            'state_stable_rotation': StableRotationVerifier(),
            'state_on_target': OnTargetVerifier(),
            'state_safe_landing': SafeLandingConditionVerifier(),
            'state_descending': DescendingVerifier(),
            'state_in_corridor': InCorridorVerifier(),
        } if use_state else {}
        
        # Weights - matching the configuration that achieved 74% success
        if use_tuned_weights:
            self.weights = {
                # Image-based (tuned to match evaluate_with_all_verifiers.py)
                'img_vertical_corridor': 2.0,   # Key differentiator
                'img_controlled_descent': 1.0,
                'img_pitch_trimming': 1.0,
                # State-based (tuned)
                'state_safe_velocity': 3.0,      # Critical for landing
                'state_stable_rotation': 2.0,    # Important for control
                'state_on_target': 3.0,          # Trajectory prediction is key
                'state_safe_landing': 2.0,       # Final approach
                'state_descending': 1.0,         # Basic sanity check
                'state_in_corridor': 2.0,        # More accurate than image version
            }
        else:
            # Equal weights
            self.weights = {name: 1.0 for name in 
                          list(self.image_verifiers.keys()) + list(self.state_verifiers.keys())}
    
    def score(self, frame_bgr: Optional[np.ndarray], obs: np.ndarray) -> Tuple[float, Dict]:
        """Score using configured verifiers."""
        results = {}
        total_score = 0.0
        
        # Run image-based verifiers
        if frame_bgr is not None:
            for name, verifier in self.image_verifiers.items():
                passed = verifier(frame_bgr)
                results[name] = bool(passed)
                if passed:
                    total_score += self.weights.get(name, 1.0)
        
        # Run state-based verifiers
        for name, verifier in self.state_verifiers.items():
            passed = verifier(obs)
            results[name] = bool(passed)
            if passed:
                total_score += self.weights.get(name, 1.0)
        
        results['total_score'] = total_score
        return total_score, results
    
    def get_verifier_names(self) -> List[str]:
        """Get list of all active verifier names."""
        return list(self.image_verifiers.keys()) + list(self.state_verifiers.keys())


def load_checkpoint(checkpoint_path: str, device: str = "mps"):
    """Load trained SDP checkpoint."""
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
    """Sample K action candidates in a single batched forward pass."""
    obs_expanded = obs_dict['obs'].expand(k, -1, -1)
    policy.action_buffer = None
    
    with torch.no_grad():
        action_dict = policy.predict_action({'obs': obs_expanded})
    
    actions = action_dict['action'].cpu().numpy()
    return [np.argmax(actions[i], axis=1) for i in range(k)]


def create_synced_env(seed: int, action_history: list):
    """Create environment synced to current state."""
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
    scorer: ConfigurableVerifierScorer,
    n_lookahead: int
) -> Tuple[float, Dict]:
    """Evaluate a candidate by simulating in a parallel environment."""
    env, obs = create_synced_env(seed, action_history)
    
    total_score = 0.0
    verifier_passes = {name: 0 for name in scorer.get_verifier_names()}
    n_frames = 0
    
    for action in candidate_actions[:n_lookahead]:
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        frame_rgb = env.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if frame_rgb is not None else None
        
        score, results = scorer.score(frame_bgr, obs)
        total_score += score
        n_frames += 1
        
        for name in scorer.get_verifier_names():
            if results.get(name, False):
                verifier_passes[name] += 1
        
        if terminated or truncated:
            if terminated and reward < 0:
                total_score -= 5.0  # Crash penalty
            break
    
    env.close()
    
    # Convert to pass rates
    verifier_pass_rates = {name: passes / max(n_frames, 1) 
                          for name, passes in verifier_passes.items()}
    
    return total_score, verifier_pass_rates


def run_single_experiment(
    policy,
    cfg,
    exp_config: ExperimentConfig,
    k: int,
    n_episodes: int,
    n_lookahead: int,
    device: str,
    seed: int,
    temperature: float,
) -> Dict:
    """Run a single experiment configuration."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    policy.set_temperature(temperature)
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    n_obs_steps = cfg.n_obs_steps
    
    scorer = ConfigurableVerifierScorer(
        use_image=exp_config.use_image_verifiers,
        use_state=exp_config.use_state_verifiers,
        use_tuned_weights=exp_config.use_tuned_weights
    )
    
    episode_rewards = []
    successes = []
    episode_times = []
    
    # Track per-verifier pass rates for successful vs failed episodes
    verifier_passes_success = {name: [] for name in scorer.get_verifier_names()}
    verifier_passes_failure = {name: [] for name in scorer.get_verifier_names()}
    
    pbar = tqdm(range(n_episodes), desc=f"{exp_config.name}", unit="ep")
    
    for ep in pbar:
        episode_start = time.time()
        episode_seed = seed + ep
        obs, _ = env.reset(seed=episode_seed)
        
        obs_history = [obs] * n_obs_steps
        policy.reset_buffer()
        
        action_history = []
        action_queue = []
        
        total_reward = 0.0
        done = False
        step = 0
        
        # Track verifier passes for this episode
        episode_verifier_passes = {name: [] for name in scorer.get_verifier_names()}
        
        while not done and step < 1000:
            if len(action_queue) > 0:
                action = action_queue.pop(0)
            else:
                obs_seq = np.stack(obs_history[-n_obs_steps:], axis=0)
                obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
                obs_dict = {'obs': obs_tensor}
                
                candidates = sample_k_candidates_batched(policy, obs_dict, k, device)
                
                if exp_config.use_verifiers and len(action_history) > 0:
                    best_idx = 0
                    best_score = -float('inf')
                    
                    for i, candidate in enumerate(candidates):
                        score, pass_rates = evaluate_candidate_parallel(
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
                    selected_actions = candidates[random.randint(0, k-1)]
                
                action_queue = list(selected_actions)
                action = action_queue.pop(0)
            
            # Execute and track verifiers on actual trajectory
            frame_rgb = env.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if frame_rgb is not None else None
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            # Score actual frame
            _, results = scorer.score(frame_bgr, obs)
            for name in scorer.get_verifier_names():
                episode_verifier_passes[name].append(results.get(name, False))
            
            obs_history.append(obs)
            action_history.append(int(action))
            total_reward += reward
            step += 1
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        success = total_reward >= 200
        successes.append(success)
        episode_rewards.append(total_reward)
        
        # Store verifier pass rates by success/failure
        for name in scorer.get_verifier_names():
            pass_rate = np.mean(episode_verifier_passes[name]) if episode_verifier_passes[name] else 0
            if success:
                verifier_passes_success[name].append(pass_rate)
            else:
                verifier_passes_failure[name].append(pass_rate)
        
        current_mean = np.mean(episode_rewards)
        current_success = np.mean(successes) * 100
        pbar.set_postfix({
            'mean': f'{current_mean:.1f}',
            'success': f'{current_success:.0f}%',
            'last': f'{total_reward:.1f}'
        })
    
    pbar.close()
    env.close()
    
    # Compute final statistics
    results = {
        'experiment_name': exp_config.name,
        'description': exp_config.description,
        'config': asdict(exp_config),
        'hyperparameters': {
            'k': k,
            'n_episodes': n_episodes,
            'n_lookahead': n_lookahead,
            'temperature': temperature,
            'seed': seed,
        },
        'metrics': {
            'success_rate': float(np.mean(successes)),
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_episode_time': float(np.mean(episode_times)),
            'total_time': float(np.sum(episode_times)),
        },
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_successes': [bool(s) for s in successes],
        'verifier_analysis': {
            'pass_rates_successful_episodes': {
                name: float(np.mean(rates)) if rates else 0.0 
                for name, rates in verifier_passes_success.items()
            },
            'pass_rates_failed_episodes': {
                name: float(np.mean(rates)) if rates else 0.0
                for name, rates in verifier_passes_failure.items()
            },
        }
    }
    
    return results


def generate_plots(all_results: List[Dict], output_dir: Path):
    """Generate all plots for the paper."""
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    names = [r['experiment_name'] for r in all_results]
    display_names = [r['description'] for r in all_results]
    success_rates = [r['metrics']['success_rate'] * 100 for r in all_results]
    mean_rewards = [r['metrics']['mean_reward'] for r in all_results]
    std_rewards = [r['metrics']['std_reward'] for r in all_results]
    times = [r['metrics']['mean_episode_time'] for r in all_results]
    
    # Colors
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    # 1. Success Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), success_rates, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_names, rotation=15, ha='right')
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Comparison Across Verifier Configurations', fontsize=14)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_comparison.png', dpi=150)
    plt.close()
    
    # 2. Reward Distribution Box Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    reward_data = [r['episode_rewards'] for r in all_results]
    bp = ax.boxplot(reward_data, labels=display_names, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Reward Distribution Across Verifier Configurations', fontsize=14)
    ax.axhline(y=200, color='green', linestyle='--', label='Success Threshold (200)')
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distribution.png', dpi=150)
    plt.close()
    
    # 3. Verifier Pass Rates Analysis (for experiments with verifiers)
    verifier_results = [r for r in all_results if r['config']['use_verifiers']]
    
    if verifier_results:
        # Get all verifier names from the hybrid_tuned experiment
        hybrid_tuned = next((r for r in all_results if r['experiment_name'] == 'hybrid_tuned'), None)
        if hybrid_tuned:
            verifier_names = list(hybrid_tuned['verifier_analysis']['pass_rates_successful_episodes'].keys())
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Success episodes
            ax1 = axes[0]
            success_data = hybrid_tuned['verifier_analysis']['pass_rates_successful_episodes']
            x = range(len(verifier_names))
            bars1 = ax1.bar(x, [success_data.get(v, 0) * 100 for v in verifier_names], color='#2ECC71')
            ax1.set_xticks(x)
            ax1.set_xticklabels([v.replace('img_', '').replace('state_', '') for v in verifier_names], 
                               rotation=45, ha='right')
            ax1.set_ylabel('Pass Rate (%)')
            ax1.set_title('Verifier Pass Rates (Successful Episodes)')
            ax1.set_ylim(0, 100)
            
            # Failed episodes
            ax2 = axes[1]
            failure_data = hybrid_tuned['verifier_analysis']['pass_rates_failed_episodes']
            bars2 = ax2.bar(x, [failure_data.get(v, 0) * 100 for v in verifier_names], color='#E74C3C')
            ax2.set_xticks(x)
            ax2.set_xticklabels([v.replace('img_', '').replace('state_', '') for v in verifier_names], 
                               rotation=45, ha='right')
            ax2.set_ylabel('Pass Rate (%)')
            ax2.set_title('Verifier Pass Rates (Failed Episodes)')
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'verifier_analysis.png', dpi=150)
            plt.close()
    
    # 4. Timing Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(names)), times, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_names, rotation=15, ha='right')
    ax.set_ylabel('Mean Episode Time (seconds)', fontsize=12)
    ax.set_title('Computational Cost Comparison', fontsize=14)
    
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_comparison.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_latex_tables(all_results: List[Dict], output_dir: Path):
    """Generate LaTeX table code."""
    
    latex = []
    latex.append("% Main Results Table")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation Study Results}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\hline")
    latex.append("Configuration & Success Rate & Mean Reward & Std Reward & Time (s) \\\\")
    latex.append("\\hline")
    
    for r in all_results:
        name = r['description']
        sr = r['metrics']['success_rate'] * 100
        mr = r['metrics']['mean_reward']
        std = r['metrics']['std_reward']
        t = r['metrics']['mean_episode_time']
        latex.append(f"{name} & {sr:.1f}\\% & {mr:.1f} & {std:.1f} & {t:.1f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:ablation}")
    latex.append("\\end{table}")
    
    # Improvement table
    baseline = next((r for r in all_results if r['experiment_name'] == 'random_baseline'), None)
    if baseline:
        latex.append("")
        latex.append("% Improvement Over Baseline Table")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Improvement Over Random Baseline}")
        latex.append("\\begin{tabular}{lcc}")
        latex.append("\\hline")
        latex.append("Configuration & Success Improvement & Reward Improvement \\\\")
        latex.append("\\hline")
        
        baseline_sr = baseline['metrics']['success_rate'] * 100
        baseline_mr = baseline['metrics']['mean_reward']
        
        for r in all_results:
            if r['experiment_name'] == 'random_baseline':
                continue
            name = r['description']
            sr_imp = r['metrics']['success_rate'] * 100 - baseline_sr
            mr_imp = r['metrics']['mean_reward'] - baseline_mr
            latex.append(f"{name} & +{sr_imp:.1f}\\% & +{mr_imp:.1f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\label{tab:improvement}")
        latex.append("\\end{table}")
    
    latex_text = "\n".join(latex)
    
    with open(output_dir / 'latex_tables.txt', 'w') as f:
        f.write(latex_text)
    
    print(f"LaTeX tables saved to {output_dir / 'latex_tables.txt'}")
    return latex_text


def main():
    parser = argparse.ArgumentParser(description="MAV-EI Ablation Study")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k', type=int, default=10, help='Number of candidates')
    parser.add_argument('--n_episodes', type=int, default=50, help='Episodes per experiment')
    parser.add_argument('--lookahead', type=int, default=16, help='Lookahead steps')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--experiments', type=str, nargs='+', 
                        default=['random_baseline', 'image_only', 'state_only', 'hybrid_equal', 'hybrid_tuned'],
                        help='Which experiments to run')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    print("Loading checkpoint...")
    policy, cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Filter experiments
    experiments_to_run = [e for e in EXPERIMENTS if e.name in args.experiments]
    
    print(f"\n{'='*60}")
    print("MAV-EI ABLATION STUDY")
    print(f"{'='*60}")
    print(f"Experiments: {[e.name for e in experiments_to_run]}")
    print(f"K: {args.k}, Episodes: {args.n_episodes}, Lookahead: {args.lookahead}")
    print(f"Temperature: {args.temperature}, Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for exp_config in experiments_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {exp_config.description}")
        print(f"{'='*60}")
        
        results = run_single_experiment(
            policy=policy,
            cfg=cfg,
            exp_config=exp_config,
            k=args.k,
            n_episodes=args.n_episodes,
            n_lookahead=args.lookahead,
            device=args.device,
            seed=args.seed,
            temperature=args.temperature,
        )
        
        all_results.append(results)
        
        # Print summary
        print(f"\n{exp_config.name} Results:")
        print(f"  Success Rate: {results['metrics']['success_rate']*100:.1f}%")
        print(f"  Mean Reward:  {results['metrics']['mean_reward']:.1f} ± {results['metrics']['std_reward']:.1f}")
        print(f"  Time/Episode: {results['metrics']['mean_episode_time']:.1f}s")
    
    # Save raw results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results, output_dir)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    latex = generate_latex_tables(all_results, output_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Configuration':<35} {'Success':>10} {'Reward':>12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['description']:<35} {r['metrics']['success_rate']*100:>9.1f}% {r['metrics']['mean_reward']:>11.1f}")
    
    # Print improvements
    baseline = next((r for r in all_results if r['experiment_name'] == 'random_baseline'), None)
    if baseline:
        print(f"\n{'Improvements over baseline:'}")
        print("-" * 60)
        for r in all_results:
            if r['experiment_name'] == 'random_baseline':
                continue
            sr_imp = r['metrics']['success_rate']*100 - baseline['metrics']['success_rate']*100
            mr_imp = r['metrics']['mean_reward'] - baseline['metrics']['mean_reward']
            print(f"{r['description']:<35} {sr_imp:>+9.1f}% {mr_imp:>+11.1f}")
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}/")
    print(f"  - ablation_results.json")
    print(f"  - success_rate_comparison.png")
    print(f"  - reward_distribution.png")
    print(f"  - verifier_analysis.png")
    print(f"  - timing_comparison.png")
    print(f"  - latex_tables.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

