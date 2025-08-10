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
    '''
    PPO Agent (on-policy, no self imitaion) for MiniGrid environments specifically designed for discrete action spaces with WandB and TensorBoard logging.
    - Actor-Critic neural network architecture
    - Clipped surrogate objective for policy updates
    - Generalized Advantage Estimation (GAE) for advantage computation
    - Experience buffer for storing and sampling transitions
    
    '''
    
    def __init__(self, env, seed=42, hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, 
                clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, batch_size=64,
                use_wandb=True, use_tensorboard=True, experiment_name=None):
        '''
        env: The environment (Minigrid)
        seed: Random seed for reproducibility
        hidden_dim: Hidden dimension size for NN
        lr: Learning rate for optimizer
        gamma: Discount factor
        lam: Lambda parameter for GAE (bias-variance tradeoff)
        clip_ratio: Clipping parameter for PPO objective
        vf_coef: Coefficient for value function loss
        ent_coef: Coefficient for entropy bonus (exploration)
        max_grad_norm: Maximum gradient norm for clipping
        batch_size: Batch size for policy updates
        '''
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.seed = seed
        
        # Environment setup
        self.env = env
        # Handle different observation space types (Box vs Discret)
        self.obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else env.observation_space.n
        self.action_dim = env.action_space.n # Discret action space
        
        # Hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # Neural network (Actor-Critic)
        self.network = PPONetwork(self.obs_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        
        # Track sample efficiency experiments 
        self.learning_curve = []  # To store (timesteps, avg_reward) pair
        self.episode_length_curve = []  # Track episodic lengths over time
        self.success_rate_curve = [] # Track success rates over time
        self.policy_behavior_curve = []  # Track policy losses and other behaviors
        
        self.episodes_completed = 0
        
        # =============== Logging setup ===============
        config = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'lambda': self.lam,
            'clip_ratio': self.clip_ratio,
            'vf_coef': self.vf_coef,
            'ent_coef': self.ent_coef,
            'max_grad_norm': self.max_grad_norm,
            'device': str(self.device)
        }
        
        self.logger = PPOLogger(
            experiment_name=experiment_name,
            use_wandb=use_wandb,
            use_tensorboard=use_tensorboard,
            env=env,
            network=self.network,
            config=config
        )
        # =============== End Logging ===============
        
        # Training metrics tracking
        self.total_timesteps = 0
        self.update_count = 0
        self.start_time = time.time()
    
    def collect_rollout(self, buffer_size=2048):
        '''
        Collect a rollout of experience from the environment.
        
        This method runs the current policy in the environment to collect
        transitions (state, action, reward, next_state, done) for training.
        '''
        # Init experience buffer
        buffer = PPOBuffer(buffer_size, self.obs_dim, self.device, 
                        gamma=self.gamma, lam=self.lam)
        # Reset env get next obeservation 
        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)
        
        # ======= Episode tracking & Logging =======
        ep_reward = 0
        ep_length = 0
        episodes_completed = 0
        action_counts = torch.zeros(self.action_dim)
        total_reward_in_rollout = 0
        # max_episode_reward = 0
        
        # Metrics for this rollout
        rollout_rewards = []
        rollout_lengths = []
        rollout_successes = []
        # ======= End =======
        
        for step in range(buffer_size): # Collect data
            # Get action from currentpolicy
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs)
            
            action_counts[action.item()] += 1 # Action distribution (Analysis)
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = terminated or truncated

            # Store transition
            buffer.store(obs, action, reward, value, log_prob, done)
            
            # Update episode tracking
            ep_reward += reward
            ep_length += 1
            total_reward_in_rollout += reward
            
            # next observation
            obs = torch.FloatTensor(next_obs).to(self.device)
            
            if done:
                success = terminated and reward >= 0.9  # Success = terminated # with positive reward
                
                # Finish trajectory in PPO buffer successfully
                buffer.finish_path()
                
                episodes_completed += 1
                self.episodes_completed += 1
                
                # Store episode stats for tracking
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
            
        # ================= Logging =================
        action_probs = action_counts / buffer_size
        success_rate = np.mean(rollout_successes) if rollout_successes else 0
        
        print(f"Action distribution: {[f'{p:.3f}' for p in action_probs.tolist()]}")
        print(f'Rollout: {episodes_completed} episodes, total_reward={total_reward_in_rollout:.1f}, '
              f'success_rate={success_rate:.3f}')
        
        # Log rollout metrics
        if rollout_rewards:
            rollout_metrics = {
                'rollout/mean_reward': np.mean(rollout_rewards),
                'rollout/success_rate': success_rate,
                'rollout/mean_length': np.mean(rollout_lengths),
                'rollout/num_episodes': len(rollout_rewards)
            }
            self.logger.log_metrics(rollout_metrics, self.total_timesteps)
        # ================= End Logging =================
        
        # Return processed batch for policy update
        return buffer.get_batch()
    
    def update_policy(self, batch, update_epochs=10, minibatch_size=64):
        '''
        Update the policy loss.
        
        This method implements the core PPO update mechanism:
        1. Compute policy and value function outputs for collected buffer
        2. Calculate PPO clipped objective
        3. Compute value function loss and entropy bonus
        4. Perform gradient ascent to improve policy
        '''
        # Extract components from batch
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
        kl_divergences = [] # KL divergence between old and new policy
        clip_fractions = []# Ratios for clipping
        
        for epoch in range(update_epochs):
            # Random minibatch sampling
            indices = torch.randperm(dataset_size)
            # Process data in minibatches
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get current policy predictions
                _, new_logp, entropy, values = self.network.get_action_and_value(
                    obs[mb_indices], act[mb_indices]
                )
                
                # PPO clipped surrogate loss
                # Ratio of new policy to old policy probabilities
                ratio = torch.exp(new_logp - old_logp[mb_indices])
                # Clipped version to prevent large policy changes
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                # PPO objective: take minimum of clipped and unclipped objectives
                policy_loss = -torch.min(
                    ratio * adv[mb_indices],
                    clipped_ratio * adv[mb_indices]
                ).mean()
                
                # Value function loss
                # MSE between predicted values and returns
                value_loss = F.mse_loss(values.squeeze(), ret[mb_indices])
                
                # Entropy loss for exploration, maximize entropy
                entropy_loss = -entropy.mean()
                
                # Total loss, combined weighted sum of losses
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad() # Clear and compute gradients
                total_loss.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step() # Update param
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(total_loss.item())
                
                # KL divergence and clip fraction
                with torch.no_grad():
                    # KL divergence between old and new policy
                    kl_div = (old_logp[mb_indices] - new_logp).mean()
                    # Fraction of probability ratios that were clipped
                    clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                    kl_divergences.append(kl_div.item())
                    clip_fractions.append(clip_frac.item())
        
        # ================== Log training metrics ==================
        training_metrics = {
            'train/policy_loss': np.mean(policy_losses),
            'train/value_loss': np.mean(value_losses),
            'train/entropy_loss': np.mean(entropy_losses),
            'train/total_loss': np.mean(total_losses),
            'train/kl_divergence': np.mean(kl_divergences),
            'train/clip_fraction': np.mean(clip_fractions),
            'train/learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.logger.log_metrics(training_metrics, self.total_timesteps)
        # ================== End Log ==================
        return {
            'policy_loss': np.mean(policy_losses),
            'total_loss': np.mean(total_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences)
        }
    
    def train(self, total_timesteps=1000000, rollout_size=2048, sil=False):
        '''
        Main training loop for the PPO agent.
        
        1. Collect experience rollouts from the environment
        2. Update the policy using collected experience
        3. Log training progress and metrics
        4. Repeat until target steps in env
        '''
        
        print('Starting PPO training...')
        print(f'Device: {self.device}')
        print(f'Observation dim: {self.obs_dim}, Action dim: {self.action_dim}')
        print(f"WandB: {'enabled' if self.logger.use_wandb else 'disabled'}")
        print(f"TensorBoard: {'enabled' if self.logger.use_tensorboard else 'disabled'}")
        #  Main training loop
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
            success_rate = np.mean(self.episode_successes) if self.episode_successes else 0
            
            # Calculate training speed
            elapsed_time = time.time() - self.start_time
            fps = self.total_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            # ======================================= LOGGING Training =======================================
            # Log general metrics
            general_metrics = {
                'general/timesteps': self.total_timesteps,
                'general/updates': self.update_count,
                'general/episodes': self.episodes_completed,
                'general/fps': fps,
                'general/avg_episode_reward': avg_reward,
                'general/avg_episode_length': avg_length,
                'general/success_rate': success_rate,
                'general/num_episodes': len(self.episode_rewards),
            }
            self.logger.log_metrics(general_metrics, self.total_timesteps)
            
            # Store learning curve data (For sample efficency)
            # Main metric to analysis sample efficiency
            self.learning_curve.append({
                'timesteps': self.total_timesteps,
                'avg_reward': avg_reward
            })
            
            self.episode_length_curve.append({
                'timesteps': self.total_timesteps,
                'avg_episode_length': avg_length
            })
            
            self.success_rate_curve.append({
                'timesteps': self.total_timesteps,
                'success_rate': success_rate
            })
            
            policy_behavior_data = {
                'timesteps': self.total_timesteps,
                'policy_loss': policy_metrics['policy_loss'],
                'total_loss': policy_metrics['total_loss'],
                'entropy_loss': policy_metrics['entropy_loss'],
                'kl_divergence': policy_metrics['kl_divergence']
            }
            
            self.policy_behavior_curve.append(policy_behavior_data)

            print(f'Update {self.update_count}')
            print(f'Timesteps: {self.total_timesteps}/{total_timesteps}')
            print(f'Episodes: {self.episodes_completed}')
            print(f'FPS: {fps:.2f}')    
            print(f'Avg Episode Reward: {avg_reward:.2f}')
            print(f'Avg Episode Length: {avg_length:.2f}')
            print(f'Success Rate: {success_rate:.2f}')
            print('-' * 50)
        
        # ======================================= END LOGGING Training =======================================

        # Close logging
        
        print('Training completed')

    # ====================== Getter methods for learning curves and training data======================
    def get_learning_curve(self):
        return self.learning_curve
    
    def get_episode_length_curve(self):
        return self.episode_length_curve
    
    def get_success_rate_curve(self):
        return self.success_rate_curve
    
    def get_policy_behavior_curve(self):
        return self.policy_behavior_curve
    # ====================== End Getter methods======================
    
    def evaluate(self, num_episodes=10, render=False, log_video=False):
        '''
        Evaluate trained policy
        
        This method runs the trained policy (without exploration)
        for a specified number of episodes to assess performance.
        '''
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        # Video logging setup for wandb
        # TODO: Fix video wandb (minigrid)
        if log_video and self.logger.use_wandb and hasattr(self.env, 'render'):
            frames = []
        
        # Similar to training loop (no gradient update)
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=self.seed + ep)
            obs = torch.FloatTensor(obs).to(self.device)
            
            ep_reward = 0
            ep_length = 0
            done = False
            
            # Run episode until completion
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
                
                # Track episode success
                if done:
                    success = terminated 
                    eval_successes.append(success)
                # Update episode statistics
                ep_reward += reward
                ep_length += 1
                
                obs = torch.FloatTensor(obs).to(self.device)
            
            eval_rewards.append(ep_reward)
            eval_lengths.append(ep_length)
            print(f'Evaluation Episode {ep + 1}: Reward = {ep_reward}, Length = {ep_length}')
        
        # Log evaluation metrics
        eval_metrics = {
            'eval/mean_reward': np.mean(eval_rewards),
            'eval/std_reward': np.std(eval_rewards),
            'eval/mean_length': np.mean(eval_lengths),
            'eval/success_rate': np.mean(eval_successes)
        }
        self.logger.log_metrics(eval_metrics, self.total_timesteps)
        
        # Log video to wandb
        if log_video and self.logger.use_wandb and 'frames' in locals() and len(frames) > 0:
            self.logger.log_video(frames)
        
        self.logger.close()
        return np.mean(eval_rewards), np.std(eval_rewards)
    
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
            }
        }, path)
        

        
        print(f'Model saved as "{path}"')
        return path