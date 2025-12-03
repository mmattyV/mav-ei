#!/usr/bin/env python3
"""
Train PPO on discrete LunarLander-v3.

Discrete action space:
  0 = do nothing
  1 = fire left engine
  2 = fire main engine
  3 = fire right engine
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback


def train_discrete_ppo(
    total_timesteps: int = 500000,
    output_dir: str = "models/ppo_discrete",
    seed: int = 42,
):
    """Train PPO on discrete Lunar Lander."""
    
    print("=" * 60)
    print("Training PPO on LunarLander-v3 (Discrete)")
    print("=" * 60)
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Output: {output_dir}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/eval", exist_ok=True)
    
    # Create training environment (4 parallel envs)
    env = make_vec_env("LunarLander-v3", n_envs=4, seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # PPO with good hyperparameters for Lunar Lander
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        tensorboard_log=f"{output_dir}/tensorboard",
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best",
        log_path=f"{output_dir}/eval",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{output_dir}/checkpoints",
        name_prefix="ppo_discrete",
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # Save final model and normalization stats
    model.save(f"{output_dir}/final_model")
    env.save(f"{output_dir}/vec_normalize.pkl")
    
    print()
    print(f"✓ Model saved to {output_dir}/final_model.zip")
    print(f"✓ Normalization saved to {output_dir}/vec_normalize.pkl")
    
    # Test
    print()
    print("Testing trained model...")
    test_model(f"{output_dir}/final_model.zip", f"{output_dir}/vec_normalize.pkl")
    
    env.close()
    eval_env.close()
    
    return model


def test_model(model_path: str, normalize_path: str, n_episodes: int = 10):
    """Test a trained model."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    model = PPO.load(model_path)
    
    env = make_vec_env("LunarLander-v3", n_envs=1, seed=999)
    if os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        
        rewards.append(total_reward)
        status = "✓" if total_reward >= 200 else "✗"
        print(f"  Episode {ep+1}: {total_reward:.2f} {status}")
    
    env.close()
    
    mean_reward = np.mean(rewards)
    print()
    print(f"Mean reward: {mean_reward:.2f} ± {np.std(rewards):.2f}")
    print(f"Success rate: {sum(r >= 200 for r in rewards)}/{n_episodes}")
    
    if mean_reward >= 200:
        print("✓ Model solved the environment!")
    
    return mean_reward


def main():
    parser = argparse.ArgumentParser(description='Train PPO on discrete Lunar Lander')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--output', type=str, default='models/ppo_discrete',
                        help='Output directory (default: models/ppo_discrete)')
    parser.add_argument('--test_only', type=str, default=None,
                        help='Path to model to test (skip training)')
    
    args = parser.parse_args()
    
    if args.test_only:
        normalize_path = os.path.join(os.path.dirname(args.test_only), "vec_normalize.pkl")
        test_model(args.test_only, normalize_path)
    else:
        train_discrete_ppo(
            total_timesteps=args.timesteps,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()

