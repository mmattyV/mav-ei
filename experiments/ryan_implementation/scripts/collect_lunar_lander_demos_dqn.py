"""
Collect expert demonstrations for LunarLander-v3 using a trained DQN model.

Usage:
    python scripts/collect_lunar_lander_demos_dqn.py \
        --weights lander_dqn_good.pth \
        --n_episodes 200 \
        --output data/lunar_lander_discrete/demonstrations.zarr
"""

import argparse
import pathlib
import sys

import numpy as np
import zarr
import gymnasium as gym
import torch
from tqdm import tqdm

# Add parent directory to path to import DQNAgent
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from dqn_agent import DQNAgent


def make_env(render_mode=None):
    """Create LunarLander-v3 environment."""
    return gym.make("LunarLander-v3", render_mode=render_mode)


def load_agent(weights_path: str, device: str | None = None) -> DQNAgent:
    """Load a trained DQN agent from weights file."""
    # Create dummy env to get dimensions
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    # Create agent and load weights
    agent = DQNAgent(state_dim, action_dim, device=device)
    state_dict = torch.load(weights_path, map_location=agent.device)
    agent.online_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(agent.online_net.state_dict())
    agent.online_net.eval()

    return agent


def collect_expert_demos(
    weights_path: str,
    n_episodes: int = 200,
    save_path: str = "data/lunar_lander_discrete/demonstrations.zarr",
    min_reward: float = 200.0,
    max_steps_per_episode: int = 1000,
):
    """
    Generate expert demonstrations using a trained DQN model.

    Args:
        weights_path: Path to trained DQN weights (.pth file)
        n_episodes: Number of successful episodes to collect
        save_path: Path to save zarr file
        min_reward: Minimum reward threshold to keep episode (200 = solved)
        max_steps_per_episode: Maximum steps per episode

    Returns:
        Path to saved zarr file, or None if collection failed
    """
    env_id = "LunarLander-v3"

    print(f"Environment: {env_id}")
    print("Action space: Discrete(4)\n")

    # Load trained DQN agent
    print(f"Loading DQN weights from: {weights_path}")
    try:
        agent = load_agent(weights_path)
        print(f"✓ Model loaded successfully (device: {agent.device})\n")
    except FileNotFoundError:
        print(f"❌ Error: Weight file not found: {weights_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Create environment
    env = make_env(render_mode=None)

    # Data collection
    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    episode_ends: list[int] = []

    successes = 0
    total_collected = 0

    print(f"Collecting demonstrations from {env_id}...")
    print(f"Target: {n_episodes} successful episodes (reward >= {min_reward})\n")

    pbar = tqdm(total=n_episodes, desc="Successful episodes")

    attempts = 0
    max_attempts = n_episodes * 5  # Prevent infinite loop

    while successes < n_episodes and attempts < max_attempts:
        attempts += 1

        state, _ = env.reset()
        episode_obs = []
        episode_actions = []
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Get greedy action from expert model
            action = agent.select_action(state, greedy=True)

            # Store obs and action BEFORE taking the step
            episode_obs.append(state.copy())
            episode_actions.append(action)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            steps += 1

        # Filter by reward threshold
        if total_reward >= min_reward:
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            episode_ends.append(len(all_obs))
            successes += 1
            pbar.update(1)
            pbar.set_postfix(
                {"reward": f"{total_reward:.1f}", "len": len(episode_actions)}
            )

        total_collected += 1

    pbar.close()
    env.close()

    if successes < n_episodes:
        print(f"\nWarning: Only collected {successes}/{n_episodes} successful episodes")
        print("Consider lowering --min_reward or using a better trained model")

    if successes == 0:
        print("\n❌ ERROR: No successful episodes collected!")
        print("Possible solutions:")
        print("  1. Lower the reward threshold: --min_reward <value>")
        print("  2. Train the model longer for better performance")
        print("  3. Check that the weights file is valid")
        return None

    print("\nStatistics:")
    print(f"  Collected: {successes} successful / {total_collected} total episodes")
    print(f"  Success rate: {100 * successes / total_collected:.1f}%")
    print(f"  Total steps: {len(all_obs)}")
    print(f"  Avg episode length: {len(all_obs) / successes:.1f}")

    # Save to zarr format
    print(f"\nSaving to {save_path}...")
    save_dir = pathlib.Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Open root group (works for zarr >= 3)
    root = zarr.open_group(save_path, mode="w")

    # Convert to numpy arrays
    obs_array = np.asarray(all_obs, dtype=np.float32)             # (T, obs_dim)
    action_array = np.asarray(all_actions, dtype=np.int64).reshape(-1, 1)  # (T, 1)
    episode_ends_array = np.asarray(episode_ends, dtype=np.int64)          # (E,)

    # Create data group
    data = root.create_group("data")
    # Use create_array instead of create_dataset to avoid the zarr v3 shape requirement
    data.create_array(
        "obs",
        data=obs_array,
        chunks=(min(1000, obs_array.shape[0]), obs_array.shape[1]),
    )
    data.create_array(
        "action",
        data=action_array,
        chunks=(min(1000, action_array.shape[0]), action_array.shape[1]),
    )

    # Create meta group
    meta = root.create_group("meta")
    meta.create_array("episode_ends", data=episode_ends_array)

    # Store metadata as attributes
    root.attrs["env_id"] = env_id
    root.attrs["n_episodes"] = int(successes)
    root.attrs["n_steps"] = int(len(all_obs))
    root.attrs["obs_dim"] = int(obs_array.shape[1])
    root.attrs["action_dim"] = 1  # discrete action stored as single int
    root.attrs["action_space"] = "discrete"
    root.attrs["n_actions"] = 4
    root.attrs["min_reward"] = float(min_reward)

    print("✓ Saved successfully!")
    print(f"  Observations: {obs_array.shape}")
    print(f"  Actions: {action_array.shape}")
    print(f"  Episodes: {len(episode_ends_array)}")

    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect Lunar Lander discrete demonstrations using a trained DQN model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained DQN weights (e.g., lander_dqn_good.pth)",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=200,
        help="Number of successful episodes to collect (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/lunar_lander_discrete/demonstrations.zarr",
        help="Output zarr file path (default: data/lunar_lander_discrete/demonstrations.zarr)",
    )
    parser.add_argument(
        "--min_reward",
        type=float,
        default=200.0,
        help="Minimum reward to keep episode (default: 200 = solved)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )

    args = parser.parse_args()

    result = collect_expert_demos(
        weights_path=args.weights,
        n_episodes=args.n_episodes,
        save_path=args.output,
        min_reward=args.min_reward,
        max_steps_per_episode=args.max_steps,
    )

    if result is None:
        print("\n⚠️  Data collection failed. Please try again with different parameters.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
