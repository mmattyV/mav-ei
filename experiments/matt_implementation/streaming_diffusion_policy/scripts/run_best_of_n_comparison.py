#!/usr/bin/env python3
"""
Run Best-of-N comparison experiments for MAV-EI.

This script runs evaluations with different K values and compares:
- K=1 (single SDP, baseline)
- K=5 (random selection)
- K=10 (random selection)
- K=20 (random selection)

The expected result is that random selection should NOT improve performance,
establishing the control baseline for verifier experiments.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluate_best_of_n_random import load_checkpoint, evaluate_best_of_n_random


def run_comparison(
    checkpoint_path: str,
    k_values: list = [1, 5, 10, 20],
    n_episodes: int = 20,
    device: str = "mps",
    seed: int = 42,
    output_dir: str = None,
):
    """
    Run Best-of-N comparison with multiple K values.
    
    Args:
        checkpoint_path: Path to SDP checkpoint
        k_values: List of K values to test
        n_episodes: Episodes per K value
        device: Device for inference
        seed: Random seed
        output_dir: Directory to save results (optional)
    """
    
    print("=" * 70)
    print("MAV-EI: Best-of-N Random Baseline Comparison")
    print("=" * 70)
    print()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"K values to test: {k_values}")
    print(f"Episodes per K: {n_episodes}")
    print(f"Device: {device}")
    print()
    
    # Load model once
    policy, cfg = load_checkpoint(checkpoint_path, device)
    
    all_results = {}
    
    for k in k_values:
        print()
        print(f"{'='*70}")
        print(f"Testing K={k}")
        print(f"{'='*70}")
        
        results = evaluate_best_of_n_random(
            policy=policy,
            cfg=cfg,
            k=k,
            n_episodes=n_episodes,
            device=device,
            render=False,
            seed=seed,
        )
        
        all_results[k] = results
        
        print(f"\n  K={k}: {results['success_rate']*100:.1f}% success, {results['mean_reward']:.1f} mean reward")
    
    # Summary comparison
    print()
    print("=" * 70)
    print("SUMMARY: Best-of-N Random Baseline Results")
    print("=" * 70)
    print()
    print(f"{'K':>4} | {'Success Rate':>12} | {'Mean Reward':>12} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 70)
    
    baseline_success = all_results[1]['success_rate'] if 1 in all_results else None
    
    for k in sorted(all_results.keys()):
        r = all_results[k]
        success_str = f"{r['success_rate']*100:.1f}%"
        
        # Compare to K=1 baseline
        if baseline_success is not None and k > 1:
            diff = (r['success_rate'] - baseline_success) * 100
            if abs(diff) < 5:
                indicator = "≈"  # Similar
            elif diff > 0:
                indicator = "↑"  # Better
            else:
                indicator = "↓"  # Worse
            success_str += f" ({indicator}{diff:+.1f})"
        
        print(f"{k:>4} | {success_str:>12} | {r['mean_reward']:>12.1f} | {r['std_reward']:>8.1f} | {r['min_reward']:>8.1f} | {r['max_reward']:>8.1f}")
    
    print()
    print("Analysis:")
    print("-" * 70)
    
    if baseline_success is not None:
        # Check if any K > 1 significantly outperforms K=1
        improvements = []
        for k in sorted(all_results.keys()):
            if k > 1:
                diff = (all_results[k]['success_rate'] - baseline_success) * 100
                improvements.append((k, diff))
        
        max_improvement = max(improvements, key=lambda x: x[1]) if improvements else (0, 0)
        
        if max_improvement[1] > 10:
            print(f"⚠️  Unexpected: K={max_improvement[0]} improved by {max_improvement[1]:.1f}%!")
            print("   This suggests sampling diversity helps, even without verifiers.")
        elif max_improvement[1] > 5:
            print(f"📊 Marginal improvement with K={max_improvement[0]} (+{max_improvement[1]:.1f}%)")
            print("   May be noise - run with more episodes to confirm.")
        else:
            print("✓ As expected: Random selection does NOT improve performance.")
            print("  This confirms that VERIFIERS (not sampling) provide value.")
    
    print()
    print("Next Steps:")
    print("-" * 70)
    print("1. Implement DQN verifier to score candidates")
    print("2. Compare Best-of-N with DQN verifier vs random")
    print("3. Expected: DQN verifier should beat random selection")
    print()
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"best_of_n_random_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON
        json_results = {}
        for k, r in all_results.items():
            json_results[str(k)] = {
                key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                for key, val in r.items()
            }
        
        with open(output_file, 'w') as f:
            json.dump({
                'checkpoint': checkpoint_path,
                'k_values': k_values,
                'n_episodes': n_episodes,
                'seed': seed,
                'results': json_results,
            }, f, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run Best-of-N comparison experiments for MAV-EI'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SDP checkpoint file')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20],
                        help='K values to test (default: 1 5 10 20)')
    parser.add_argument('--n_episodes', type=int, default=20,
                        help='Episodes per K value (default: 20)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (mps/cpu) (default: mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='data/experiments',
                        help='Directory to save results (default: data/experiments)')
    
    args = parser.parse_args()
    
    run_comparison(
        checkpoint_path=args.checkpoint,
        k_values=args.k_values,
        n_episodes=args.n_episodes,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

