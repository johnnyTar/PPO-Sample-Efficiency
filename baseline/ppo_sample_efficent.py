import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper
import wandb
import os
import time
from datetime import datetime

class PPONetwork(nn.Module):
    """Neural network for PPO with shared layers and separate heads for actor/critic"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared_features = self.shared(x)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value
    
    def get_action_and_value(self, x, action=None):
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, size, obs_dim, device):
        self.size = size
        self.obs_dim = obs_dim
        self.device = device
        self.reset()
        
    def reset(self):
        self.observations = torch.zeros((self.size, self.obs_dim), device=self.device)
        self.actions = torch.zeros(self.size, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(self.size, device=self.device)
        self.values = torch.zeros(self.size, device=self.device)
        self.log_probs = torch.zeros(self.size, device=self.device)
        self.dones = torch.zeros(self.size, device=self.device)
        self.ptr = 0
        self.path_start_idx = 0
        
    def store(self, obs, act, rew, val, log_prob, done):
        assert self.ptr < self.size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        
    def finish_path(self, last_val=0):
        """Compute GAE-Lambda advantages and returns"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat([self.rewards[path_slice], torch.tensor([last_val], device=self.device)])
        vals = torch.cat([self.values[path_slice], torch.tensor([last_val], device=self.device)])
        
        # GAE-Lambda advantage estimation
        deltas = rews[:-1] + 0.99 * vals[1:] - vals[:-1]
        advantages = torch.zeros_like(deltas)
        advantage = 0
        
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + 0.99 * 0.95 * advantage
            advantages[t] = advantage
            
        # Returns = advantages + values
        returns = advantages + vals[:-1]
        
        # Store advantages and returns
        if not hasattr(self, 'advantages'):
            self.advantages = torch.zeros(self.size, device=self.device)
            self.returns = torch.zeros(self.size, device=self.device)
            
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
        
    def get_batch(self):
        """Get all data with advantages normalized"""
        assert self.ptr == self.size
        self.ptr = 0
        self.path_start_idx = 0
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std
        
        return {
            'obs': self.observations,
            'act': self.actions,
            'ret': self.returns,
            'adv': self.advantages,
            'logp': self.log_probs
        }

class PPOAgent:
    """PPO Agent for MiniGrid environments with WandB and TensorBoard logging"""
    
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, 
                 clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 use_wandb=True, use_tensorboard=True, experiment_name=None):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # Neural network
        self.network = PPONetwork(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Logging setup
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"PPO_MiniGrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        
        # Initialize logging
        self._setup_logging()
        
        # Training metrics tracking
        self.total_timesteps = 0
        self.update_count = 0
        self.start_time = time.time()
        
    def _setup_logging(self):
        """Setup WandB and TensorBoard logging"""
        
        # WandB setup
        if self.use_wandb:
            wandb.init(
                project="ppo-minigrid",
                name=self.experiment_name,
                config={
                    "algorithm": "PPO",
                    "environment": str(self.env.spec.id) if hasattr(self.env, 'spec') else "MiniGrid",
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
            )
            
            # Watch the model
            wandb.watch(self.network, log="all", log_freq=100)
        
        # TensorBoard setup
        if self.use_tensorboard:
            log_dir = f"runs/{self.experiment_name}"
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
            # Log hyperparameters
            hparams = {
                "gamma": self.gamma,
                "lambda": self.lam,
                "clip_ratio": self.clip_ratio,
                "vf_coef": self.vf_coef,
                "ent_coef": self.ent_coef,
                "max_grad_norm": self.max_grad_norm
            }
            self.writer.add_hparams(hparams, {})
    
    def _log_metrics(self, metrics_dict):
        """Log metrics to both WandB and TensorBoard"""
        
        if self.use_wandb:
            wandb.log(metrics_dict)
        
        if self.use_tensorboard:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, self.total_timesteps)
    
    def collect_rollout(self, buffer_size=2048):
        """Collect rollout data"""
        buffer = PPOBuffer(buffer_size, self.obs_dim, self.device)
        
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
        if not done:
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
            self._log_metrics(rollout_metrics)
        
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
        self._log_metrics(training_metrics)
    
    def train(self, total_timesteps=1000000, rollout_size=2048, log_interval=10):
        """Main training loop"""
        
        print("Starting PPO training...")
        print(f"Device: {self.device}")
        print(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        print(f"WandB: {'enabled' if self.use_wandb else 'disabled'}")
        print(f"TensorBoard: {'enabled' if self.use_tensorboard else 'disabled'}")
        
        while self.total_timesteps < total_timesteps:
            # Collect rollout
            batch = self.collect_rollout(rollout_size)
            self.total_timesteps += rollout_size
            
            # Update policy
            self.update_policy(batch)
            self.update_count += 1
            
            # Logging and progress tracking
            #if self.update_count % log_interval == 0:
            # print(f"Reward {self.episode_rewards} Length {self.episode_lengths}")
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
            self._log_metrics(general_metrics)
            
            print(f"Update {self.update_count}")
            print(f"Timesteps: {self.total_timesteps}/{total_timesteps}")
            print(f"FPS: {fps:.0f}")
            print(f"Avg Episode Reward: {avg_reward:.2f}")
            print(f"Avg Episode Length: {avg_length:.2f}")
            print("-" * 50)
        
        # Close logging
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.writer.close()
    
    def evaluate(self, num_episodes=10, render=False, log_video=False):
        """Evaluate trained policy"""
        eval_rewards = []
        eval_lengths = []
        
        # Video logging setup for wandb
        if log_video and self.use_wandb and hasattr(self.env, 'render'):
            frames = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            obs = torch.FloatTensor(obs).to(self.device)
            
            ep_reward = 0
            ep_length = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Capture frames for video logging
                if log_video and self.use_wandb and ep == 0:  # Only record first episode
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
        self._log_metrics(eval_metrics)
        
        # Log video to wandb
        if log_video and self.use_wandb and 'frames' in locals() and len(frames) > 0:
            wandb.log({"eval/video": wandb.Video(np.array(frames), fps=4, format="gif")})
        
        return np.mean(eval_rewards), np.std(eval_rewards)
    
    def save_model(self, path=None):
        """Save the trained model"""
        if path is None:
            path = f"{self.experiment_name}_model.pth"
        
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
        
        # Also log to wandb if enabled
        if self.use_wandb:
            wandb.save(path)
        
        print(f"Model saved as '{path}'")
        return path
from gymnasium.wrappers import RecordVideo, TimeLimit
# Example usage
if __name__ == "__main__":
    import random
    seed = 42

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    # Create MiniGrid environment
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = FlatObsWrapper(env)  # Flatten observation for easier handling
    env.reset(seed=seed)
    run_name = 'Test_Run_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    video_dir = os.path.join('videos', run_name)
    os.makedirs(video_dir, exist_ok=True)
    
    env = RecordVideo(
                env,
                f'videos/{run_name}',
                episode_trigger=lambda ep: ep % 10 == 0,  # Record every 10 episodes
                video_length=200
            )
    
    # Create PPO agent with logging
    agent = PPOAgent(
        env, 
        hidden_dim=256, 
        lr=3e-4,
        use_wandb=True,
        use_tensorboard=True,
        experiment_name="PPO_MiniGrid_Example"
    )
    
    # Train the agent
    agent.train(total_timesteps=500000, rollout_size=2048, log_interval=5)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = agent.evaluate(num_episodes=10, log_video=False)
    print(f"Mean evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save the trained model
    model_path = agent.save_model()
    print(f"Training completed! Check your logs:")
    if agent.use_wandb:
        print(f"- WandB: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
    if agent.use_tensorboard:
        print(f"- TensorBoard: tensorboard --logdir runs/{agent.experiment_name}")