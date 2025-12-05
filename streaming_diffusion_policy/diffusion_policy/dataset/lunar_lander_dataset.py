"""
Lunar Lander dataset for Diffusion Policy training.
"""

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class LunarLanderDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=16,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.1,
            max_train_episodes=None
            ):
        """
        Lunar Lander dataset for training Diffusion Policy.
        
        Args:
            zarr_path: Path to demonstrations.zarr file
            horizon: Length of action sequence to predict
            pad_before: Number of steps to pad before sequence
            pad_after: Number of steps to pad after sequence
            seed: Random seed for train/val split
            val_ratio: Fraction of episodes for validation
            max_train_episodes: Maximum number of training episodes (None = use all)
        """
        super().__init__()
        
        # Load replay buffer from zarr
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['obs', 'action'])
        
        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        # Create sequence sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
        )
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """Return validation dataset with same configuration."""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Compute normalizer for observations and actions.
        
        For discrete actions (one-hot encoded), we scale from [0,1] to [-1,1]
        since diffusion models work best with data in [-1,1] range.
        
        Args:
            mode: Normalization mode ('limits' or 'gaussian')
        """
        import torch
        actions = self.replay_buffer['action']
        
        # Check if actions are one-hot encoded (discrete)
        # One-hot: values in {0, 1} and sum per row = 1
        is_onehot = (
            np.allclose(actions.min(), 0) and 
            np.allclose(actions.max(), 1) and
            np.allclose(actions.sum(axis=1), 1)
        )
        
        normalizer = LinearNormalizer()
        
        if is_onehot:
            # For one-hot actions: 
            # - Normalize obs to [-1, 1] (standard)
            # - Scale actions from [0,1] to [-1,1] for diffusion
            
            # Fit obs normalizer
            obs_data = {'obs': self.replay_buffer['obs']}
            normalizer.fit(data=obs_data, last_n_dims=1, mode=mode, **kwargs)
            
            # Add action normalizer that maps [0,1] -> [-1,1]
            action_data = {'action': actions}
            action_normalizer = LinearNormalizer()
            action_normalizer.fit(data=action_data, last_n_dims=1, mode=mode, **kwargs)
            normalizer.params_dict['action'] = action_normalizer.params_dict['action']
            
            # Override to scale [0,1] -> [-1,1]
            # Formula: output = input * scale + offset
            # We want: 0 -> -1, 1 -> 1
            # So: scale = 2, offset = -1
            action_dim = actions.shape[-1]
            normalizer.params_dict['action']['scale'].data = torch.ones(action_dim) * 2.0
            normalizer.params_dict['action']['offset'].data = torch.ones(action_dim) * (-1.0)
            
            print("INFO: Detected one-hot actions - scaling [0,1] -> [-1,1] for diffusion")
        else:
            # For continuous actions: normalize everything
            data = {
                'obs': self.replay_buffer['obs'],
                'action': actions
            }
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions in dataset."""
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            dict with keys:
                'obs': [horizon, obs_dim] - observation sequence
                'action': [horizon, action_dim] - action sequence
        """
        sample = self.sampler.sample_sequence(idx)
        
        # Lunar Lander obs: [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]
        # Lunar Lander action (continuous): [main_engine, lateral_engine]
        data = {
            'obs': sample['obs'].astype(np.float32),
            'action': sample['action'].astype(np.float32),
        }
        
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data



