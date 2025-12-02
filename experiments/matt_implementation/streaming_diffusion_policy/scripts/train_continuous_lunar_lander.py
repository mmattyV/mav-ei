#!/usr/bin/env python3
"""
Train a continuous PPO model for LunarLanderContinuous-v3.
This creates a high-quality expert policy for demonstration collection.

Usage:
    python scripts/train_continuous_lunar_lander.py
    
Training takes ~20-30 minutes on CPU, ~5-10 minutes on GPU.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
import os


def train_continuous_ppo(
    total_timesteps: int = 500000,
    save_path: str = "models/lunar_lander_continuous",
    eval_freq: int = 10000,
    n_eval_episodes: int = 20,
):
    """
    Train PPO on LunarLanderContinuous-v3.
    
    Args:
        total_timesteps: Total training steps (500k = ~30 min CPU, 10 min GPU)
        save_path: Directory to save the trained model
        eval_freq: Evaluate every N steps
        n_eval_episodes: Number of episodes for evaluation
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/eval", exist_ok=True)
    
    print("=" * 60)
    print("Training Continuous PPO for LunarLanderContinuous-v3")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save path: {save_path}")
    print(f"Eval frequency: {eval_freq:,}")
    print()
    
    # Create vectorized training environment (8 parallel environments)
    print("Creating training environments...")
    train_env = make_vec_env(
        "LunarLanderContinuous-v3",
        n_envs=8,
        seed=42
    )
    
    # Normalize observations and rewards for stable training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        "LunarLanderContinuous-v3",
        n_envs=4,
        seed=123
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during eval
        clip_obs=10.0,
        training=False,
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=f"{save_path}/eval",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{save_path}/checkpoints",
        name_prefix="ppo_lunar_continuous",
        verbose=1,
    )
    
    # Create PPO model with good hyperparameters for LunarLander
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,           # Collect 1024 steps per env before update
        batch_size=64,          # Minibatch size
        n_epochs=4,             # Number of epochs per update
        gamma=0.999,            # Discount factor
        gae_lambda=0.98,        # GAE lambda
        clip_range=0.2,         # PPO clip range
        ent_coef=0.01,          # Entropy coefficient (encourages exploration)
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Gradient clipping
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard",
        seed=42,
    )
    
    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print("Progress will be shown below.")
    print("Monitor with: tensorboard --logdir models/lunar_lander_continuous/tensorboard")
    print()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # Save final model
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    final_path = f"{save_path}/final_model"
    model.save(final_path)
    train_env.save(f"{save_path}/vec_normalize.pkl")
    
    print(f"✓ Final model saved to: {final_path}.zip")
    print(f"✓ Best model saved to: {save_path}/best/best_model.zip")
    print(f"✓ Normalization stats saved to: {save_path}/vec_normalize.pkl")
    
    # Test the trained model
    print()
    print("Testing trained model...")
    normalize_path = f"{save_path}/vec_normalize.pkl"
    test_model(final_path, normalize_path, n_episodes=10)
    
    return final_path


def test_model(model_path: str, normalize_path: str, n_episodes: int = 10):
    """Test a trained model with proper normalization."""
    
    # Load model
    model = PPO.load(model_path)
    
    # Create test environment with normalization
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = make_vec_env("LunarLanderContinuous-v3", n_envs=1, seed=999)
    env = VecNormalize.load(normalize_path, env)
    env.training = False  # Don't update stats during testing
    env.norm_reward = False  # Don't normalize rewards during testing
    
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # VecEnv returns array
            done = done[0]  # VecEnv returns array
        
        rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.2f}")
    
    env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print()
    print(f"Test Results:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Success rate: {sum(r >= 200 for r in rewards)}/{n_episodes} episodes")
    
    if mean_reward >= 200:
        print("  ✓ Model solved the environment! (reward >= 200)")
    else:
        print(f"  ⚠️  Model needs more training (current: {mean_reward:.2f}, target: 200)")
    
    return mean_reward


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train continuous PPO for LunarLander')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500k)')
    parser.add_argument('--output', type=str, 
                        default='models/lunar_lander_continuous',
                        help='Output directory for models')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--test_only', type=str, default=None,
                        help='Only test an existing model (provide model path without .zip)')
    
    args = parser.parse_args()
    
    if args.test_only:
        print(f"Testing model: {args.test_only}")
        # Assume vec_normalize.pkl is in the parent directory of the model
        # e.g., models/lunar_lander_continuous/final_model -> models/lunar_lander_continuous/vec_normalize.pkl
        import os
        model_dir = os.path.dirname(args.test_only)
        normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
        if not os.path.exists(normalize_path):
            # Try one level up
            normalize_path = os.path.join(os.path.dirname(model_dir), "vec_normalize.pkl")
        test_model(args.test_only, normalize_path, n_episodes=20)
    else:
        train_continuous_ppo(
            total_timesteps=args.timesteps,
            save_path=args.output,
            eval_freq=args.eval_freq,
        )

