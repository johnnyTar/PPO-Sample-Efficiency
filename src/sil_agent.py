import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from agent import PPOAgent
from sil_buffer import SILBuffer
from buffer import PPOBuffer

class PPOSILAgent(PPOAgent):
    '''
    PPO with Self-Imitation Learning (SIL) Agent Implementation

    This module extends the base PPO agent with Self-Imitation Learning capabilities
    to improve sample efficiency in sparse reward environments. SIL allows the agent
    to learn from its own past successful experiences by storing good episodes in a
    replay buffer and periodically training on them.

    Key Features:
    - Inherits all PPO functionality from base PPOAgent
    - Self-Imitation Learning with prioritized experience replay
    - Adaptive success thresholds for episode selection
    - Separate SIL optimizer with different learning rate

    SIL Algorithm Overview:
    1. Collect experience using current policy (same as PPO)
    2. Store successful episodes in SIL buffer based on return threshold
    3. Perform standard PPO updates on recent experience
    4. Additionally perform SIL updates on stored successful experiences
    5. SIL updates use clipped policy gradients on positive advantages only
    '''
    
    def __init__(self, env, seed=42, hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, 
                clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, batch_size=64,
                use_wandb=True, use_tensorboard=True, experiment_name=None,
                # SIL specific parameters
                sil=True, sil_lr=1e-4, sil_buffer_size=10000, sil_batch_size=32,
                sil_update_ratio=2, sil_alpha=0.6, sil_beta=0.4, sil_clip_ratio=None,
                success_threshold=0.8, sil_warmup_episodes=50, sil_update_frequency=5,):
        '''
        Same as parent class
        SIL-specific parameters:
            sil: enable SIL
            sil_lr: Learning rate for SIL optimizer (lower than main Lr)
            sil_buffer_size: Maximum number of episodes in SIL buffer
            sil_batch_size: Batch size for SIL updates
            sil_update_ratio: Number of SIL updates per PPO update
            sil_alpha: Prioritized replay alpha parameter: priority
            sil_beta: same as alpha: importance sampling
            sil_clip_ratio: Clipping ratio for SIL policy updates
            success_threshold: Minimum return threshold for storing episodes in SIL buffer
            sil_warmup_episodes: Number of episodes before starting SIL updates
            sil_update_frequency (int): Frequency of SIL updates (every N PPO updates)
        '''
        # Initialize base PPO agent
        super().__init__(env, seed, hidden_dim, lr, gamma, lam, clip_ratio, vf_coef, 
                        ent_coef, max_grad_norm, batch_size, use_wandb, use_tensorboard, 
                        experiment_name)
        
        # SIL parameters
        self.sil = sil
        self.sil_lr = sil_lr
        self.sil_batch_size = sil_batch_size
        self.sil_update_ratio = sil_update_ratio
        self.sil_clip_ratio = sil_clip_ratio if sil_clip_ratio is not None else clip_ratio
        self.sil_warmup_episodes = sil_warmup_episodes
        self.sil_update_frequency = sil_update_frequency
        
        # SIL optimizer and buffer

        self.sil_optimizer = optim.Adam(self.network.parameters(), lr=sil_lr)
        self.sil_buffer = SILBuffer(
            capacity=sil_buffer_size,
            device=self.device,
            alpha=sil_alpha,
            beta=sil_beta,
            success_threshold=success_threshold,
        )
        
        # SIL tracking
        self.sil_losses = []
        self.sil_updates = 0
        
        # Update logger config with SIL parameters
        sil_config = {
            'sil': self.sil,
            'sil_lr': self.sil_lr if self.sil else None,
            'sil_buffer_size': sil_buffer_size if self.sil else None,
            'success_threshold': success_threshold if self.sil else None,
        }
        self.logger.config.update(sil_config)
    
    def collect_rollout(self, buffer_size=2048):
        '''
        Collect rollout data similar to without SIL and store only successful episodes in SIL buffer.
        
        This method extends the base PPO rollout collection by storing experience 
        in the SIL buffer (off-policy). Episodes that meet the success
        criteria (based on return threshold) are stored for later SIL updates.
        
        The method maintains two buffers:
        1. PPO buffer: Recent experience for standard PPO updates(on-policy)
        2. SIL buffer: Successful episodes for self-imitation learning (off-policy)
        '''
        # Initialize PPO buffer
        buffer = PPOBuffer(buffer_size, self.obs_dim, self.device, 
                        gamma=self.gamma, lam=self.lam)
        # Reset environment and get initial observation
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        # tracking variables
        ep_reward = 0
        ep_length = 0
        episodes_completed = 0
        action_counts = torch.zeros(self.action_dim)
        total_reward_in_rollout = 0
        
        # Rollout metrics
        rollout_rewards = []
        rollout_lengths = []
        rollout_successes = []
        
        # Start episode in SIL buffer
        self.sil_buffer.start_episode()
        for step in range(buffer_size):
            # Get action from policy
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs)
            
            action_counts[action.item()] += 1
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            # Store transition in PPO buffer
            buffer.store(obs, action, reward, value, log_prob, done)
            
            # Store transition in SIL buffer (for current episode)
            self.sil_buffer.add_transition(
                obs, action, reward, log_prob, value,
            )
            
            # Update episode tracking
            ep_reward += reward
            ep_length += 1
            total_reward_in_rollout += reward
            
            # Move to next observation
            obs = torch.FloatTensor(next_obs).to(self.device)
            
            if done:
                success = terminated and reward >= 0.9  # Success = terminated with positive reward
                
                # Finish trajectory in PPO buffer
                buffer.finish_path()
                
                # Finish episode in SIL buffer
                # This evaluates if episode should be stored based on return threshold
                self.sil_buffer.finish_episode(ep_reward, self.gamma)
                self.sil_buffer.start_episode()  # Start new episode
                
                # Update episode counters
                episodes_completed += 1
                self.episodes_completed += 1
                
                # Store episode stats
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_successes.append(success)
                rollout_rewards.append(ep_reward)
                rollout_lengths.append(ep_length)
                rollout_successes.append(success)
                
                # Reset environment
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)
                ep_reward = 0
                ep_length = 0
        
        # Handle final trajectory if not done
        if buffer.path_start_idx < buffer.ptr:
            with torch.no_grad():
                _, _, _, last_val = self.network.get_action_and_value(obs)
            buffer.finish_path(last_val.item())
            
        # =============================== Logging ===============================
        action_probs = action_counts / buffer_size
        success_rate = np.mean(rollout_successes) if rollout_successes else 0
        
        print(f"Action distribution: {[f'{p:.3f}' for p in action_probs.tolist()]}")
        print(f'Rollout: {episodes_completed} episodes, total_reward={total_reward_in_rollout:.1f}, '
              f'success_rate={success_rate:.3f}')
        
        # Log SIL buffer stats
        sil_stats = self.sil_buffer.get_stats()
        print(f"SIL Buffer: size={sil_stats['buffer_size']}, "
              f"episodes_added={sil_stats['episodes_added']}, "
              f"mean_return={sil_stats.get('mean_return', 0):.2f}")
        
        # Log rollout metrics
        rollout_metrics = {}
        if rollout_rewards:
            rollout_metrics = {
                'rollout/mean_reward': np.mean(rollout_rewards),
                'rollout/success_rate': success_rate,
                'rollout/mean_length': np.mean(rollout_lengths),
                'rollout/num_episodes': len(rollout_rewards)
            }
            
        # Add SIL metrics
        if sil_stats['buffer_size'] > 0:
            rollout_metrics.update({
                'sil/buffer_size': sil_stats['buffer_size'],
                'sil/episodes_added': sil_stats['episodes_added'],
                'sil/buffer_mean_return': sil_stats.get('mean_return', 0),
                'sil/beta': sil_stats.get('beta', 0.4)
            })
        
        self.logger.log_metrics(rollout_metrics, self.total_timesteps)
        # =============================== End Logging ===============================
        
        return buffer.get_batch()
    
    def update_sil(self):
        
        '''
        Perform Self-Imitation Learning updates using stored successful experiences.
        
        1. Sampling episodes from the SIL buffer (prioritized by advantages)
        2. Computing advantages using current value function
        3. Performing policy updates only on transitions with positive advantages
        4. Using clipped policy gradients similar to PPO but without entropy regularization
        
        The key insight is that we only want to imitate actions that led to better
        outcomes than the current value function predicts, the focus on positive advantages.
        '''
        # Check if SIL updates should be performed
        if not self.sil or len(self.sil_buffer) < self.sil_batch_size:
            # print(f'======================1{self.sil_batch_size}')
            return {'sil_loss': 0.0, 'sil_policy_loss': 0.0, 'sil_value_loss': 0.0}
        
        # Only update SIL after warmup and at specified frequency
        if (self.episodes_completed < self.sil_warmup_episodes or 
            self.update_count % self.sil_update_frequency != 0):
            # print(f'======================2{self.episodes_completed}')
            return {'sil_loss': 0.0, 'sil_policy_loss': 0.0, 'sil_value_loss': 0.0}
        # loss tracking
        sil_losses = []
        sil_policy_losses = []
        sil_value_losses = []
        
        # Perform multiple SIL updates
        for _ in range(self.sil_update_ratio):
            # Sample batch from SIL buffer using prioritized replay
            sil_batch = self.sil_buffer.sample(self.sil_batch_size)
            if sil_batch is None:
                continue
                
            # Extract batch component
            obs = sil_batch['obs']
            actions = sil_batch['act']  
            returns = sil_batch['ret'] # Monte Carlo returns
            old_log_probs = sil_batch['logp'] # Log prob uding old policy
            importance_weights = sil_batch['weights'] # Importance sampling weights
            indices = sil_batch['indices']
            
            # Get current policy predictions
            _, new_log_probs, _, new_values = self.network.get_action_and_value(obs, actions)
            
            # Compute advantages using current value function
            advantages = returns - new_values.squeeze()
            
            # Only learn from transitions with positive advantages
            # that mean we only imitate actions that led to better outcomes than
            # what the current value function predicts
            positive_advantages = torch.clamp(advantages, min=0.0)
            
            # Skip update if no positive advantages
            if positive_advantages.sum() == 0:
                continue
            
            # Create mask for positive advantages only
            positive_mask = advantages > 0
            if not positive_mask.any():
                continue
            
            # Apply mask to positive
            masked_advantages = positive_advantages[positive_mask]
            masked_new_log_probs = new_log_probs[positive_mask]
            masked_old_log_probs = old_log_probs[positive_mask]
            # The data in SIL buffer was collected with an old policy, 
            # but updating with the current policy. This is off-policy learning.
            masked_weights = importance_weights[positive_mask]
            masked_new_values = new_values.squeeze()[positive_mask]
            masked_returns = returns[positive_mask]
            
            # Skip if no valid transitions after masking
            if len(masked_advantages) == 0:
                continue
                
            # Policy loss with clipping (only positive)
            ratio = torch.exp(masked_new_log_probs - masked_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.sil_clip_ratio, 1 + self.sil_clip_ratio)
            
            # manimize log prob weighted by positive advantages
            policy_loss = -torch.min(
                ratio * masked_advantages,
                clipped_ratio * masked_advantages
            )
            # Weight by importance sampling weights from prioritized replay
            # Weight each sample by how representative it is
            policy_loss = (policy_loss * masked_weights).mean()
            
            # Value loss 
            value_loss = F.mse_loss(masked_new_values, masked_returns, reduction='none')
            value_loss = (value_loss * masked_weights).mean()
            
            # Total SIL loss (no entropy regularization)
            sil_loss = policy_loss + 0.5 * value_loss
            
            # Perform optimization step only if loss is valid
            if not torch.isnan(sil_loss) and sil_loss.item() > 0:
                # Optimization step (similar to ppo agent)
                self.sil_optimizer.zero_grad()
                sil_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.sil_optimizer.step()
                
                # Track metrics
                sil_losses.append(sil_loss.item())
                sil_policy_losses.append(policy_loss.item())
                sil_value_losses.append(value_loss.item())
                
                # Update priorities in SIL buffer based on new advantages
                # Higher advantages get higher priority for future sampling
                with torch.no_grad():
                    positive_indices = indices[positive_mask.cpu().numpy()]
                    new_priorities = torch.abs(masked_advantages).cpu().numpy()
                    self.sil_buffer.update_priorities(positive_indices, new_priorities)
        
        self.sil_updates += 1
        
        return {
            'sil_loss': np.mean(sil_losses) if sil_losses else 0.0,
            'sil_policy_loss': np.mean(sil_policy_losses) if sil_policy_losses else 0.0,
            'sil_value_loss': np.mean(sil_value_losses) if sil_value_losses else 0.0
        }
    
    def update_policy(self, batch, update_epochs=10, minibatch_size=64):
        '''
        Update policy using both PPO and SIL
        
        1. First perform standard PPO updates on recent experience
        2. Then perform SIL updates on stored successful experiences
        '''
        
        # Perform base PPO update
        ppo_metrics = super().update_policy(batch, update_epochs, minibatch_size)
        
        # Perform SIL updates after PPO updates
        sil_metrics = self.update_sil()
        
        # Add SIL metrics to training metrics
        if self.sil:
            sil_training_metrics = {
                'sil/loss': sil_metrics['sil_loss'],
                'sil/policy_loss': sil_metrics['sil_policy_loss'],
                'sil/value_loss': sil_metrics['sil_value_loss'],
                'sil/updates': self.sil_updates,
                'sil/learning_rate': self.sil_optimizer.param_groups[0]['lr']
            }
            self.logger.log_metrics(sil_training_metrics, self.total_timesteps)

        # Combine metrics
        combined_metrics = ppo_metrics.copy()
        combined_metrics.update({
            'sil_loss': sil_metrics['sil_loss'],
            'sil_policy_loss': sil_metrics['sil_policy_loss']
        })
        if combined_metrics["sil_loss"] > 0:
            print(f'SIL Loss: {combined_metrics["sil_loss"]:.4f}')
        
        return combined_metrics
    
    def train(self, total_timesteps=1000000, rollout_size=2048, sil=True):
        '''Main training loop for the PPO agent.'''
        # This methode is same as PPO agent, just add other logging
        print('Starting Improved PPO with SIL training...')
        print(f'Device: {self.device}')
        print(f'Observation dim: {self.obs_dim}, Action dim: {self.action_dim}')
        print(f'SIL enabled: {self.sil}')
        # SIL logging
        print(f'SIL warmup episodes: {self.sil_warmup_episodes}')
        print(f'SIL update frequency: every {self.sil_update_frequency} PPO updates')
        sil_stats = self.sil_buffer.get_stats()
        print(f'Success threshold: {sil_stats.get("current_threshold", "N/A")}')
        
        print(f"WandB: {'enabled' if self.logger.use_wandb else 'disabled'}")
        print(f"TensorBoard: {'enabled' if self.logger.use_tensorboard else 'disabled'}")
        
        # Call train with SIL-specific
        super().train(total_timesteps, rollout_size, sil=sil)
    
    def save_model(self, path=None):
        '''Save the trained model'''
        if path is None:
            path = f'{self.logger.experiment_name}_model.pth'
        
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_ratio': self.clip_ratio,
                'vf_coef': self.vf_coef,
                'ent_coef': self.ent_coef
            },
            'sil': {
                'sil_lr': self.sil_lr,
                'sil_batch_size': self.sil_batch_size,
                'sil_update_ratio': self.sil_update_ratio,
                'sil_warmup_episodes': self.sil_warmup_episodes
            }
        }, path)
        
        
        
        print(f'Model saved as "{path}"')
        return path
    
    