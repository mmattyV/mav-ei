#!/usr/bin/env python3
"""
Visualize trained PPO model playing Lunar Lander.

This opens a window showing the agent in action.
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import time
import os


def visualize_ppo(
    model_path: str,
    normalize_path: str,
    n_episodes: int = 5,
    delay: float = 0.02,  # Slow down for visibility
):
    """
    Visualize the PPO agent playing Lunar Lander.
    """
    print("=" * 60)
    print("PPO Lunar Lander Visualization")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print()
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    def make_env():
        return gym.make("LunarLanderContinuous-v3", render_mode="human")
    
    # For VecNormalize, we need a VecEnv
    env = DummyVecEnv([make_env])
    
    # Load normalization if available
    if os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
        print("✓ Loaded VecNormalize stats")
    else:
        print("⚠️  No VecNormalize found, using raw environment")
    
    print()
    print("Starting visualization... (close window to stop)")
    print()
    
    rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = done[0]
            
            # Slow down for visibility
            time.sleep(delay)
        
        success = "✓ LANDED!" if total_reward >= 200 else "✗ crashed"
        print(f"  Episode {ep+1}: reward = {total_reward:.2f}, steps = {steps} {success}")
        rewards.append(total_reward)
    
    env.close()
    
    print()
    print("=" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Success Rate: {sum(r >= 200 for r in rewards)}/{n_episodes}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize PPO on Lunar Lander')
    parser.add_argument('--model', type=str, 
                        default='models/lunar_lander_continuous/final_model.zip',
                        help='Path to PPO model')
    parser.add_argument('--normalize', type=str,
                        default='models/lunar_lander_continuous/vec_normalize.pkl',
                        help='Path to VecNormalize stats')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--delay', type=float, default=0.02,
                        help='Delay between steps (seconds)')
    
    args = parser.parse_args()
    
    visualize_ppo(
        model_path=args.model,
        normalize_path=args.normalize,
        n_episodes=args.episodes,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

