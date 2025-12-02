"""
Lunar Lander environment runner for policy evaluation.
"""

import gymnasium as gym
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import time

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class LunarLanderRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            n_train=8,
            n_train_vis=2,
            train_start_seed=0,
            n_test=20,
            n_test_vis=4,
            test_start_seed=10000,
            max_steps=1000,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            fps=50,
            crf=22,
            continuous=True,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Environment runner for Lunar Lander evaluation.
        
        Args:
            output_dir: Directory for saving outputs
            n_train: Number of training environments
            n_train_vis: Number of training videos to record
            train_start_seed: Starting seed for training envs
            n_test: Number of test environments
            n_test_vis: Number of test videos to record
            test_start_seed: Starting seed for test envs
            max_steps: Maximum steps per episode
            n_obs_steps: Observation history length
            n_action_steps: Number of actions to predict
            n_latency_steps: Simulated latency steps
            fps: Frames per second for video recording
            crf: Video compression quality (lower = better)
            continuous: Use continuous action space
            past_action: Include past actions in observation
            tqdm_interval_sec: Progress bar update interval
            n_envs: Number of parallel environments (None = n_train + n_test)
        """
        super().__init__(output_dir)
        
        if n_envs is None:
            n_envs = n_train + n_test
        
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps
        
        def env_fn():
            env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    gym.make(env_id, render_mode="rgb_array"),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )
        
        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        
        # Train environments
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis
            
            def init_fn(env, seed=seed, enable_render=enable_render):
                # Setup video recording
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f'train_ep_{seed}.mp4')
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                
                # Set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        # Test environments
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis
            
            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f'test_ep_{seed}.mp4')
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)
                
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy):
        """
        Run policy evaluation across all environments.
        
        Args:
            policy: Policy to evaluate
            
        Returns:
            dict: Evaluation metrics including rewards and videos
        """
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_prediction_times = []
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            
            # Init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])
            
            # Start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            
            pbar = tqdm.tqdm(total=self.max_steps, 
                desc=f"Eval LunarLander {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # Create obs dict
                np_obs_dict = {
                    'obs': obs[..., :self.n_obs_steps, :].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps-1):].astype(np.float32)
                
                # Device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))
                
                # Run policy
                with torch.no_grad():
                    start_predict = time.time()
                    action_dict = policy.predict_action(obs_dict)
                    end_predict = time.time()
                    all_prediction_times.append(end_predict - start_predict)
                
                # Device transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                
                action = np_action_dict['action'][:, self.n_latency_steps:]
                
                # Step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action
                
                pbar.update(action.shape[1])
            pbar.close()
            
            # Collect results
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        
        # Log results
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video
        
        # Log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value
        
        log_data['mean_prediction_time'] = np.mean(all_prediction_times)
        
        return log_data


