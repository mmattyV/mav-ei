"""
Collect expert demonstrations for LunarLanderContinuous-v3 using a trained PPO model.

Usage:
    python scripts/collect_lunar_lander_demos.py \
        --model models/lunar_lander_continuous/final_model.zip \
        --n_episodes 200 \
        --output data/lunar_lander/demonstrations.zarr
"""

import argparse
import pathlib
import numpy as np
import zarr
import gymnasium as gym
from tqdm import tqdm

def collect_expert_demos(
    model_path,
    n_episodes=200,
    save_path='data/lunar_lander/demonstrations.zarr',
    min_reward=200,
):
    """
    Generate expert demonstrations using a trained PPO model.
    
    Args:
        model_path: Path to trained PPO model (required)
        n_episodes: Number of episodes to collect
        save_path: Path to save zarr file
        min_reward: Minimum reward threshold to keep episode (200 = solved)
    """
    
    # Setup environment (continuous control only)
    env_id = "LunarLanderContinuous-v3"
    
    print(f"Environment: {env_id}")
    print(f"Action space: Box(-1.0, 1.0, (2,), float32)")
    print()
    
    # Load trained model with normalization
    print(f"Loading model from: {model_path}")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from stable_baselines3.common.env_util import make_vec_env
        import os
        
        # Load model
        model = PPO.load(model_path)
        
        # Load normalization stats (should be in same directory as model)
        model_dir = os.path.dirname(model_path)
        normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
        
        # Create vectorized environment with normalization
        env = make_vec_env(env_id, n_envs=1, seed=42)
        
        if os.path.exists(normalize_path):
            env = VecNormalize.load(normalize_path, env)
            env.training = False  # Don't update stats during collection
            env.norm_reward = False  # Don't normalize rewards
            print("✓ Model and normalization loaded successfully")
        else:
            print("⚠️  Warning: vec_normalize.pkl not found, using unnormalized environment")
            print(f"   Expected at: {normalize_path}")
        
        print()
    except ImportError as e:
        print(f"❌ Error: Missing dependency - {e}")
        print("Install with: pip install stable-baselines3")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Data collection
    all_obs = []
    all_actions = []
    episode_ends = []
    
    successes = 0
    total_collected = 0
    
    print(f"\nCollecting demonstrations from {env_id}...")
    print(f"Target: {n_episodes} successful episodes (reward > {min_reward})")
    
    pbar = tqdm(total=n_episodes, desc="Successful episodes")
    
    attempts = 0
    max_attempts = n_episodes * 5  # Prevent infinite loop
    
    while successes < n_episodes and attempts < max_attempts:
        attempts += 1
        
        obs = env.reset()
        episode_obs = [obs[0]]  # VecEnv returns array of observations
        episode_actions = []
        done = False
        total_reward = 0
        
        while not done:
            # Get action from expert model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # VecEnv returns arrays
            total_reward += reward[0]
            done = done[0]
            
            episode_obs.append(obs[0])
            episode_actions.append(action[0])
        
        # Filter by reward threshold
        if total_reward >= min_reward:
            all_obs.extend(episode_obs[:-1])  # Don't include terminal state
            all_actions.extend(episode_actions)
            episode_ends.append(len(all_obs))
            successes += 1
            pbar.update(1)
            pbar.set_postfix({'reward': f'{total_reward:.1f}', 'len': len(episode_actions)})
        
        total_collected += 1
    
    pbar.close()
    
    if successes < n_episodes:
        print(f"\nWarning: Only collected {successes}/{n_episodes} successful episodes")
        print(f"Consider lowering --min_reward or training the model longer")
    
    # Check if we collected any successful episodes
    if successes == 0:
        print("\n❌ ERROR: No successful episodes collected!")
        print("Possible solutions:")
        print("  1. Lower the reward threshold: --min_reward <value>")
        print("  2. Train the model longer for better performance")
        print("  3. Use the best checkpoint: models/lunar_lander_continuous/best/best_model.zip")
        env.close()
        return None
    
    print(f"\nStatistics:")
    print(f"  Collected: {successes} successful / {total_collected} total episodes")
    print(f"  Success rate: {100*successes/total_collected:.1f}%")
    print(f"  Total steps: {len(all_obs)}")
    print(f"  Avg episode length: {len(all_obs)/successes:.1f}")
    
    # Save to zarr format
    print(f"\nSaving to {save_path}...")
    save_dir = pathlib.Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    root = zarr.open(save_path, 'w')
    
    # Convert to numpy arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    action_array = np.array(all_actions, dtype=np.float32)
    episode_ends_array = np.array(episode_ends, dtype=np.int64)
    
    # Create data group (required by ReplayBuffer)
    data = root.create_group('data')
    data.create_dataset('obs', data=obs_array, chunks=(1000, obs_array.shape[1]))
    data.create_dataset('action', data=action_array, chunks=(1000, action_array.shape[1]))
    
    # Create meta group
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends', data=episode_ends_array)
    
    # Store metadata
    root.attrs['env_id'] = env_id
    root.attrs['n_episodes'] = successes
    root.attrs['n_steps'] = len(all_obs)
    root.attrs['obs_dim'] = obs_array.shape[1]
    root.attrs['action_dim'] = action_array.shape[1]
    root.attrs['min_reward'] = min_reward
    
    print(f"✓ Saved successfully!")
    print(f"  Observations: {obs_array.shape}")
    print(f"  Actions: {action_array.shape}")
    print(f"  Episodes: {len(episode_ends_array)}")
    
    env.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description='Collect Lunar Lander continuous demonstrations using a trained PPO model'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (e.g., models/lunar_lander_continuous/final_model.zip)')
    parser.add_argument('--n_episodes', type=int, default=200,
                        help='Number of successful episodes to collect (default: 200)')
    parser.add_argument('--output', type=str, 
                        default='data/lunar_lander/demonstrations.zarr',
                        help='Output zarr file path (default: data/lunar_lander/demonstrations.zarr)')
    parser.add_argument('--min_reward', type=float, default=200,
                        help='Minimum reward to keep episode (default: 200 = solved)')
    
    args = parser.parse_args()
    
    result = collect_expert_demos(
        model_path=args.model,
        n_episodes=args.n_episodes,
        save_path=args.output,
        min_reward=args.min_reward,
    )
    
    if result is None:
        print("\n⚠️  Data collection failed. Please try again with different parameters.")
        exit(1)


if __name__ == "__main__":
    main()


