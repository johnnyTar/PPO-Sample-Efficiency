import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from collections import deque

from network import PPONetwork
from buffer import PPOBuffer
from logger import PPOLogger


class PPOAgent:
    """PPO Agent for MiniGrid environments with WandB and TensorBoard logging"""
    
    def __init__(self, env, seed=42, hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, 
                clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, batch_size=64,
                use_wandb=True, use_tensorboard=True, experiment_name=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # Environment setup
        self.env = env
        self.obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.n
        self.action_dim = env.action_space.n
        
        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # Neural network
        self.network = PPONetwork(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        # Track sample efficiency experiments 
        self.learning_curve = []  # To store (timesteps, avg_reward) pair
        self.episode_length_curve = []  # Track episodic lengths over time
        self.policy_behavior_curve = []  # Track policy losses and other behaviors
        
        # Logging setup
        config = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "lambda": self.lam,
            "clip_ratio": self.clip_ratio,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "device": str(self.device)
        }
        
        self.logger = PPOLogger(
            experiment_name=experiment_name,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            env=env,
            network=self.network,
            config=config
        )
        
        # Training metrics tracking
        self.total_timesteps = 0
        self.update_count = 0
        self.start_time = time.time()
    
    def collect_rollout(self, buffer_size=2048):
        """Collect rollout data"""
        buffer = PPOBuffer(buffer_size, self.obs_dim, self.device, 
                        gamma=self.gamma, lam=self.lam)
        
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        
        ep_reward = 0
        ep_length = 0
        
        # Metrics for this rollout
        rollout_rewards = []
        rollout_lengths = []
        
        for step in range(buffer_size):
            # Get action from policy
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            # Store transition
            buffer.store(obs, action, reward, value, log_prob, done)
            
            # Update episode tracking
            ep_reward += reward
            ep_length += 1
            
            # Move to next observation
            obs = torch.FloatTensor(next_obs).to(self.device)
            
            if done:
                # Finish the trajectory
                buffer.finish_path()
                
                # Store episode stats
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                rollout_rewards.append(ep_reward)
                rollout_lengths.append(ep_length)
                
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
        
        # Log rollout metrics
        if rollout_rewards:
            rollout_metrics = {
                "rollout/mean_reward": np.mean(rollout_rewards),
                "rollout/max_reward": np.max(rollout_rewards),
                "rollout/min_reward": np.min(rollout_rewards),
                "rollout/mean_length": np.mean(rollout_lengths),
                "rollout/num_episodes": len(rollout_rewards)
            }
            self.logger.log_metrics(rollout_metrics, self.total_timesteps)
        
        return buffer.get_batch()
    
    def update_policy(self, batch, update_epochs=10, minibatch_size=64):
        """Update policy using PPO loss"""
        
        obs = batch['obs']
        act = batch['act']
        ret = batch['ret']
        adv = batch['adv']
        old_logp = batch['logp']
        
        dataset_size = obs.shape[0]
        
        # Track losses for logging
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        kl_divergences = []
        clip_fractions = []
        
        for epoch in range(update_epochs):
            # Random minibatch sampling
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get current policy predictions
                _, new_logp, entropy, values = self.network.get_action_and_value(
                    obs[mb_indices], act[mb_indices]
                )
                
                # PPO clipped surrogate loss
                ratio = torch.exp(new_logp - old_logp[mb_indices])
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss = -torch.min(
                    ratio * adv[mb_indices],
                    clipped_ratio * adv[mb_indices]
                ).mean()
                
                # Value function loss
                value_loss = F.mse_loss(values.squeeze(), ret[mb_indices])
                
                # Entropy loss for exploration
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(total_loss.item())
                
                # KL divergence and clip fraction
                with torch.no_grad():
                    kl_div = (old_logp[mb_indices] - new_logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                    kl_divergences.append(kl_div.item())
                    clip_fractions.append(clip_frac.item())
        
        # Log training metrics
        training_metrics = {
            "train/policy_loss": np.mean(policy_losses),
            "train/value_loss": np.mean(value_losses),
            "train/entropy_loss": np.mean(entropy_losses),
            "train/total_loss": np.mean(total_losses),
            "train/kl_divergence": np.mean(kl_divergences),
            "train/clip_fraction": np.mean(clip_fractions),
            "train/learning_rate": self.optimizer.param_groups[0]['lr']
        }
        self.logger.log_metrics(training_metrics, self.total_timesteps)

        return {
            'policy_loss': np.mean(policy_losses),
            'total_loss': np.mean(total_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences)
        }
    
    def train(self, total_timesteps=1000000, rollout_size=2048, log_interval=10):
        """Main training loop"""
        
        print("Starting PPO training...")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"WandB: {'enabled' if self.logger.use_wandb else 'disabled'}")
        print(f"TensorBoard: {'enabled' if self.logger.use_tensorboard else 'disabled'}")
        
        while self.total_timesteps < total_timesteps:
            # Collect rollout
            batch = self.collect_rollout(rollout_size)
            self.total_timesteps += rollout_size
            
            # Update policy
            policy_metrics = self.update_policy(batch)
            self.update_count += 1
            
            # Logging and progress tracking
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            
            # Calculate training speed
            elapsed_time = time.time() - self.start_time
            fps = self.total_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            # Log general metrics
            general_metrics = {
                "general/timesteps": self.total_timesteps,
                "general/updates": self.update_count,
                "general/fps": fps,
                "general/avg_episode_reward": avg_reward,
                "general/avg_episode_length": avg_length,
                "general/num_episodes": len(self.episode_rewards)
            }
            self.logger.log_metrics(general_metrics, self.total_timesteps)
            
            # Store learning curve data  (For sample efficency)
            self.learning_curve.append({
                "timesteps": self.total_timesteps,
                "avg_reward": avg_reward
            })
            
            # Store additional curves
            self.episode_length_curve.append({
                "timesteps": self.total_timesteps,
                "avg_episode_length": avg_length
            })

            self.policy_behavior_curve.append({
                "timesteps": self.total_timesteps,
                "policy_loss": policy_metrics['policy_loss'],
                "total_loss": policy_metrics['total_loss'],
                "entropy_loss": policy_metrics['entropy_loss'],
                "kl_divergence": policy_metrics['kl_divergence']
            })

            print(f"Update {self.update_count}")
            print(f"Timesteps: {self.total_timesteps}/{total_timesteps}")
            print(f"Avg Episode Reward: {avg_reward:.2f}")
            print(f"Avg Episode Length: {avg_length:.2f}")
            print("-" * 50)
        
        # Close logging
        
        print("Training completed")

    def get_learning_curve(self):
        return self.learning_curve
    
    def get_episode_length_curve(self):
        return self.episode_length_curve
    
    def get_policy_behavior_curve(self):
        return self.policy_behavior_curve
    
    def evaluate(self, num_episodes=10, render=False, log_video=False):
        """Evaluate trained policy"""
        eval_rewards = []
        eval_lengths = []
        
        # Video logging setup for wandb
        if log_video and self.logger.use_wandb and hasattr(self.env, 'render'):
            frames = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed + ep)
            obs = torch.FloatTensor(obs).to(self.device)
            
            ep_reward = 0
            ep_length = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Capture frames for video logging
                if log_video and self.logger.use_wandb and ep == 0:  # Only record first episode
                    if hasattr(self.env, 'render'):
                        frame = self.env.render(mode='rgb_array')
                        if frame is not None:
                            frames.append(frame)
                
                with torch.no_grad():
                    action, _, _, _ = self.network.get_action_and_value(obs)
                
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
                
                obs = torch.FloatTensor(obs).to(self.device)
            
            eval_rewards.append(ep_reward)
            eval_lengths.append(ep_length)
            print(f"Evaluation Episode {ep + 1}: Reward = {ep_reward}, Length = {ep_length}")
        
        # Log evaluation metrics
        eval_metrics = {
            "eval/mean_reward": np.mean(eval_rewards),
            "eval/std_reward": np.std(eval_rewards),
            "eval/max_reward": np.max(eval_rewards),
            "eval/min_reward": np.min(eval_rewards),
            "eval/mean_length": np.mean(eval_lengths)
        }
        self.logger.log_metrics(eval_metrics, self.total_timesteps)
        
        # Log video to wandb
        if log_video and self.logger.use_wandb and 'frames' in locals() and len(frames) > 0:
            self.logger.log_video(frames)
        
        self.logger.close()
        return np.mean(eval_rewards), np.std(eval_rewards)
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = f"{self.logger.experiment_name}_model.pth"
        
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
            }
        }, path)
        

        
        print(f"Model saved as '{path}'")
        return path