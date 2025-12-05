# python main.py --mode train --train-episodes 1000 --weights lander_dqn_good.pth
# python main.py --mode train --train-episodes 20 --weights lander_dqn_sandbag.pth

# python main.py --mode watch --weights lander_dqn_good.pth --episodes 3
# python main.py --mode watch --weights lander_dqn_sandbag.pth --episodes 3

# python main.py --mode benchmark --weights lander_dqn_good.pth --episodes 100

import argparse
import os
import random
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import cv2
import torch

from dqn_agent import DQNAgent


# Directory for screenshots used by image-based verifiers
SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)


def make_env(render_mode=None):
    """
    Helper to make LunarLander-v3 with the given render mode.
    render_mode:
        - None        : no rendering (fast training)
        - "human"     : pygame window
        - "rgb_array" : returns frames as numpy arrays from env.render()
    """
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    return env

def maybe_save_screenshot(ep_idx: int, shot_idx: int, frame_rgb: np.ndarray) -> None:
    """
    Save a *given* RGB frame for episode ep_idx and logical shot index shot_idx.
    frame_rgb should be shape (H, W, 3) in RGB.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    filename = SCREENSHOT_DIR / f"ep{ep_idx:03d}_shot_{shot_idx:02d}.png"
    cv2.imwrite(str(filename), frame_bgr)
    print(f"[screenshot] saved {filename}")


def train(
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    seed: int = 0,
):
    """
    Training loop wired specifically to this DQNAgent:

      - agent.select_action(state)
      - agent.store_transition(s, a, r, s', done)
      - agent.update(global_step)
    """
    env = make_env(render_mode=None)
    env.action_space.seed(seed)

    state, _ = env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    scores = []
    scores_window = deque(maxlen=100)

    global_step = 0
    start_time = time.perf_counter()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for t in range(max_steps_per_episode):
            # epsilon-greedy handled inside select_action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # store experience in replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # one gradient step (if buffer is warm enough)
            agent.update(global_step)
            global_step += 1

            state = next_state
            episode_reward += reward

            if done:
                break

        scores.append(episode_reward)
        scores_window.append(episode_reward)

        avg_last_100 = np.mean(scores_window)
        eps_val = agent.epsilon()

        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:7.2f} | "
            f"Average(100): {avg_last_100:7.2f} | "
            f"Epsilon: {eps_val:.3f}"
        )

    elapsed = time.perf_counter() - start_time
    print(f"Training finished in {elapsed:.1f} seconds (wall clock).")

    env.close()
    return agent, scores

def evaluate(
    agent: DQNAgent,
    num_episodes: int = 5,
    take_screenshots: bool = False,
    max_steps_per_episode: int = 1000,
    shots_per_episode: int = 10,
):
    """
    Evaluate the agent.

    If take_screenshots=True, we:
      - record EVERY frame for the episode as an RGB array
      - after the episode ends, choose `shots_per_episode` frames
        that are evenly spaced over the *usable* part of the
        trajectory (we skip an initial prefix of frames so the
        lander is fully in view).
    """
    env = make_env(render_mode="human")
    scores = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        frames = []  # will store RGB frames for this episode

        while not done and steps < max_steps_per_episode:
            # Greedy action for evaluation (no exploration).
            action = agent.select_action(state, greedy=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Grab the current pygame display as an RGB frame
            if take_screenshots:
                surf = pygame.display.get_surface()
                if surf is not None:
                    arr = pygame.surfarray.array3d(surf)  # (W, H, 3) RGB
                    frame_rgb = np.transpose(arr, (1, 0, 2))  # -> (H, W, 3)
                    frames.append(frame_rgb)

            state = next_state
            steps += 1

        scores.append(episode_reward)
        print(f"Eval Episode {ep + 1}: reward = {episode_reward:.2f}")

        # After the episode ends, pick evenly spaced frames to save
        if take_screenshots and len(frames) > 0:
            # Skip the first 10% of frames to avoid "spawn" views
            skip_frac = 0.10
            usable_start = int(len(frames) * skip_frac)

            # Make sure we still have something left
            if usable_start >= len(frames):
                usable_start = 0

            usable_frames = frames[usable_start:]
            if len(usable_frames) == 0:
                usable_frames = frames  # fallback

            num_shots = min(shots_per_episode, len(usable_frames))
            indices = np.linspace(0, len(usable_frames) - 1,
                                  num=num_shots, dtype=int)

            for shot_idx, frame_idx in enumerate(indices):
                frame_rgb = usable_frames[frame_idx]
                maybe_save_screenshot(ep_idx=ep, shot_idx=shot_idx, frame_rgb=frame_rgb)

    env.close()
    return scores


def benchmark(
    agent: DQNAgent,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    success_reward_threshold: float = 200.0,
):
    """
    Run many episodes without rendering and report success rate.
    
    A landing is considered "successful" if episode reward >= success_reward_threshold.
    LunarLander gives +100 for landing on the pad, so 200+ usually means a good landing.
    """
    env = make_env(render_mode=None)
    
    successes = 0
    rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state, greedy=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            steps += 1
        
        rewards.append(episode_reward)
        if episode_reward >= success_reward_threshold:
            successes += 1
        
        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{num_episodes} | Reward: {episode_reward:.2f}")
    
    env.close()
    
    success_rate = successes / num_episodes * 100
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print("\n" + "=" * 50)
    print(f"Benchmark Results ({num_episodes} episodes)")
    print("=" * 50)
    print(f"Success Rate:    {successes}/{num_episodes} ({success_rate:.1f}%)")
    print(f"Average Reward:  {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward:      {np.min(rewards):.2f}")
    print(f"Max Reward:      {np.max(rewards):.2f}")
    print("=" * 50)
    
    return {
        "success_rate": success_rate,
        "successes": successes,
        "total": num_episodes,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "rewards": rewards,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DQN LunarLander main")
    parser.add_argument(
        "--mode",
        choices=["train", "watch", "benchmark"],
        default="train",
        help="train: train a new agent; watch: run episodes with rendering; benchmark: run many episodes and report success rate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="number of episodes for watch / benchmark mode",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=1000,
        help="number of training episodes (when mode=train)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="lander_dqn.pth",
        help="path to save/load model weights",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=200.0,
        help="reward threshold to consider a landing successful (for benchmark mode)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        # Train from scratch
        agent, scores = train(num_episodes=args.train_episodes, seed=args.seed)

        # Save weights (online network)
        torch.save(agent.online_net.state_dict(), args.weights)
        print(f"Saved weights to {args.weights}")

        # Quick evaluation run without rendering (optional sanity check)
        print("Running quick evaluation (no rendering)...")
        env = make_env(render_mode=None)
        for i in range(5):
            state, _ = env.reset()
            done = False
            total_r = 0.0
            steps = 0
            while not done and steps < 1000:
                action = agent.select_action(state, greedy=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_r += reward
                state = next_state
                steps += 1
            print(f"Eval Episode {i+1}: reward = {total_r:.2f}")
        env.close()

    elif args.mode == "watch":
        # Watch mode: load weights if available, otherwise init fresh agent.
        env = make_env(render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        agent = DQNAgent(state_dim, action_dim)

        if os.path.exists(args.weights):
            state_dict = torch.load(args.weights, map_location=agent.device)
            agent.online_net.load_state_dict(state_dict)
            agent.target_net.load_state_dict(agent.online_net.state_dict())
            print(f"Loaded weights from {args.weights}")
        else:
            print(
                f"WARNING: Weight file {args.weights} not found. "
                "Using randomly initialized agent."
            )

        evaluate(agent, num_episodes=args.episodes, take_screenshots=True)

    elif args.mode == "benchmark":
        # Load weights and run benchmark
        env = make_env(render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()

        agent = DQNAgent(state_dim, action_dim)

        if os.path.exists(args.weights):
            state_dict = torch.load(args.weights, map_location=agent.device)
            agent.online_net.load_state_dict(state_dict)
            agent.target_net.load_state_dict(agent.online_net.state_dict())
            print(f"Loaded weights from {args.weights}")
        else:
            print(
                f"ERROR: Weight file {args.weights} not found. "
                "Cannot run benchmark without trained weights."
            )
            return

        benchmark(
            agent,
            num_episodes=args.episodes,
            success_reward_threshold=args.success_threshold,
        )

    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
