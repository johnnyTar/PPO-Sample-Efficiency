import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from collections import deque

from network import PPONetwork
from rnd_buffer import PPORNDBuffer
from logger import PPOLogger
from agent import PPOAgent

from rnd import RNDModel, RunningMeanStd


class PPORNDAgent(PPOAgent):
    '''PPO 
    PPO Agent for reinforcement learning with optional RND intrinsic rewards.

    This module extend the PPO algorithm to include RND for sample efficiency.
    RND (Random Network Distillation) is used to provide intrinsic rewards based 
    on the novelty of states.

    Key Features:
    - Implements Proximal Policy Optimization (PPO) algorithm.
    - Supports RND for intrinsic rewards to improve sample efficiency.


    RND Algorithm:
    1. **Target Network**: A fixed network that generates target embeddings for observations.
    2. **Predictor Network**: A trainable network that tries to predict the target embeddings.
    3. **Intrinsic Reward**: computed as the prediction error between the predictor and target networks.
    4. **Normalization**: Both observations and intrinsic.
    5. **Integration**: combine intrinsic rewards with extrinsic rewards.
    6. **Training**: PPO updates the policy using the combined rewards.


    '''
    
    def __init__(self, env, seed=42, hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, 
                clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, batch_size=64,
                use_wandb=True, use_tensorboard=True, experiment_name=None,
                # RND parameters
                use_rnd=False, intrinsic_coef=0.5, rnd_embed_dim=128, rnd_lr=1e-4,
                rnd_obs_norm=True, rnd_rew_norm=True, rnd_update_epochs=1):
        '''
        same as PPOAgent, but with additional RND parameters.
        RND parameters:
        - use_rnd (bool): Whether to use RND for intrinsic rewards.
        - intrinsic_coef (float): Coefficient for intrinsic reward scaling.
        - rnd_embed_dim (int): Dimension of the RND embeddings.
        - rnd_lr (float): Learning rate for the RND predictor.
        - rnd_obs_norm (bool): Whether to normalize observations for RND.
        - rnd_rew_norm (bool): Whether to normalize intrinsic rewards.
        - rnd_update_epochs (int): Number of epochs to update the RND predictor per PPO update.
        '''
        
        # Initialize base PPO agent
        super().__init__(env, seed, hidden_dim, lr, gamma, lam, clip_ratio, vf_coef, 
                        ent_coef, max_grad_norm, batch_size, use_wandb, use_tensorboard, 
                        experiment_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(self.device)
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
        self.episode_successes = deque(maxlen=100)
        
        # Track sample efficiency experiments 
        self.learning_curve = []  # To store (timesteps, avg_reward) pair
        self.episode_length_curve = []  # Track episodic lengths over time
        self.success_rate_curve = [] # Track success rates over time
        self.policy_behavior_curve = []  # Track policy losses and other behaviors
        
        # Logging setup
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
        
        # Training metrics tracking
        self.total_timesteps = 0
        self.update_count = 0
        self.start_time = time.time()

        # RND setup
        self.use_rnd = use_rnd
        self.intrinsic_coef = intrinsic_coef
        self.rnd_update_epochs = rnd_update_epochs
        self.episode_ext_rewards = deque(maxlen=100)
        self.episode_int_rewards = deque(maxlen=100)
        self.episode_comb_rewards = deque(maxlen=100)

        if self.use_rnd:
            self.rnd = RNDModel(self.obs_dim, embed_dim=rnd_embed_dim, hidden_dim=hidden_dim).to(self.device)
            self.rnd_opt = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
            self.rnd_obs_norm = rnd_obs_norm
            self.rnd_rew_norm = rnd_rew_norm
            self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
            self.int_rms = RunningMeanStd(shape=())

        config.update({ 'use_rnd': self.use_rnd,
                       'intrinsic_coef': self.intrinsic_coef 
                       })

    def collect_rollout(self, buffer_size=2048):
        '''
        Collect rollout data
        
        This method collects a rollout of interactions with the environment.
        It gathers observations, actions, rewards, and other necessary data
        to fill the PPO buffer for training.
        Args:
            buffer_size (int): Number of steps to collect in the rollout.
        Returns:
            batch (dict): A dictionary containing the collected data.
        '''
        buffer = PPORNDBuffer(buffer_size, self.obs_dim, self.device, 
                        gamma=self.gamma, lam=self.lam)
        
        obs, _ = self.env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        ep_ext_reward = 0.0
        ep_int_reward = 0.0
        ep_length = 0
        rollout_ext_rewards = []
        rollout_int_rewards = []
        rollout_comb_rewards = []
        rollout_lengths = []

        action_counts = torch.zeros(self.action_dim)
        
        
        for step in range(buffer_size):
            # Get action from policy
            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(obs_t)
            
            action_counts[action.item()] += 1
            # Take step in environment
            next_obs, ext_rew, terminated, truncated, _ = self.env.step(int(action.item()))
            # if ext_rew > 0:
            #     print(f"Step {ep_length}: Got reward {ext_rew:.2f}. Terminated={terminated}, Truncated={truncated}")

            is_true_terminal = bool(terminated)   # environment terminal (done by env)
            is_truncated = bool(truncated)        # time-limit truncation

            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)


            # Intrinsic reward on next_obs
            int_rew = 0.0
            if self.use_rnd:
                no = next_obs.copy()
                if self.rnd_obs_norm:
                    self.obs_rms.update(no[None, ...])
                    no = self.obs_rms.normalize(no)
                no_t = torch.as_tensor(no, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    pred, tgt = self.rnd(no_t.unsqueeze(0).to(self.device))
                    int_err = F.mse_loss(pred, tgt, reduction='none').mean(dim=1)
                int_rew = int_err.item()
                if self.rnd_rew_norm:
                    self.int_rms.update(np.array([int_rew], dtype=np.float32))
                    int_rew = int_rew / np.sqrt(self.int_rms.var + 1e-8)

            # If the environment truly terminated, zero intrinsic reward for that transition
            if is_true_terminal:
                int_rew = 0.0


            combined_rew = float(ext_rew + self.intrinsic_coef * int_rew)

            # Store transition with combined reward
            # store done flag = 1.0 only for true termination (terminated), not for truncation
            buffer.store(obs_t, action, combined_rew, value, log_prob, float(is_true_terminal), next_obs_t)
            
            # Update episode tracking
            ep_ext_reward += ext_rew
            ep_int_reward += int_rew
            ep_length += 1

            obs_t = next_obs_t
            
            
            done = is_true_terminal or is_truncated
            if done:
                # If the episode was truncated, bootstrap with the critic's value prediction.
                # If it terminated, the future reward is zero.
                if is_truncated:
                    with torch.no_grad():
                        _, _, _, v_boot = self.network.get_action_and_value(next_obs_t)
                    buffer.finish_path(last_val=float(v_boot.item()))
                else:  # is_true_terminal
                    buffer.finish_path(last_val=0.0)

                
                # Record episode statistics
                self.episode_rewards.append(ep_ext_reward)
                self.episode_ext_rewards.append(ep_ext_reward)
                self.episode_lengths.append(ep_length)
                self.episode_successes.append(bool(is_true_terminal)) # This will now correctly log successes

                # (Add any other stat recording here, like for RND rewards)
                if self.use_rnd:
                    self.episode_int_rewards.append(ep_int_reward)
                    self.episode_comb_rewards.append(ep_ext_reward + self.intrinsic_coef * ep_int_reward)

                # Add to the rollout-specific lists for logging
                rollout_ext_rewards.append(ep_ext_reward)
                rollout_int_rewards.append(ep_int_reward)
                rollout_comb_rewards.append(ep_ext_reward + self.intrinsic_coef * ep_int_reward)
                rollout_lengths.append(ep_length)

                # Reset environment for the next episode
                obs, _ = self.env.reset()
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

                # Reset episode accumulators
                ep_ext_reward = 0.0
                ep_int_reward = 0.0
                ep_length = 0
        
        # Handle final trajectory if buffer has unfinished path (bootstrap with critic)
        if buffer.path_start_idx < buffer.ptr:
            with torch.no_grad():
                _, _, _, last_val = self.network.get_action_and_value(obs_t)
            buffer.finish_path(last_val=float(last_val.item()))
            
        # logging
        action_probs = (action_counts / buffer_size).tolist()
        print(f"Action distribution: {[f'{p:.3f}' for p in action_probs]}")
        # print(f'Max episode reward: {max_episode_reward}')
        
        # Log rollout metrics
        if rollout_ext_rewards:
            self.logger.log_metrics({
                'rollout/mean_extrinsic_reward': float(np.mean(rollout_ext_rewards)),
                'rollout/mean_intrinsic_reward': float(np.mean(rollout_int_rewards)) if self.use_rnd else 0.0,
                'rollout/mean_combined_reward': float(np.mean(rollout_comb_rewards)),
                'rollout/mean_length': float(np.mean(rollout_lengths)),
                'rollout/num_episodes': len(rollout_ext_rewards)
            }, self.total_timesteps)

        return buffer.get_batch()
    
    def update_policy(self, batch, update_epochs=10, minibatch_size=64):
        '''
        Update policy using PPO loss

        This method performs the PPO update step using the collected batch of data.
        It computes the policy loss, value function loss, and entropy loss,
        and updates the policy network accordingly.
        Args:
            batch (dict): A dictionary containing the collected data.
            update_epochs (int): Number of epochs to run the PPO update.
            minibatch_size (int): Size of the minibatches for training.
        Returns:
            dict: A dictionary containing the training metrics.

        '''
        
        obs = batch['obs']
        next_obs = batch['next_obs']
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
        rnd_losses = []
        rnd_pred_errs = []
        
        for epoch in range(update_epochs):
            # Random minibatch sampling
            indices = torch.randperm(dataset_size, device=self.device)
            
            for start in range(0, dataset_size, minibatch_size):
                end = start + minibatch_size
                mb = indices[start:end]

                _, new_logp, entropy, values = self.network.get_action_and_value(
                    obs[mb], act[mb]
                )
                
                # PPO clipped surrogate loss
                ratio = torch.exp(new_logp - old_logp[mb])
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * adv[mb], clipped_ratio * adv[mb]).mean()
        
                
                # Value function loss
                value_loss = F.mse_loss(values.squeeze(), ret[mb])
                
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
                    kl_div = (old_logp[mb] - new_logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
                    kl_divergences.append(kl_div.item())
                    clip_fractions.append(clip_frac.item())
                
                # RND predictor update
                if self.use_rnd:
                    x = next_obs[mb]
                    if self.rnd_obs_norm:
                        x_np = x.detach().cpu().numpy()
                        self.obs_rms.update(x_np)
                        x = torch.as_tensor(self.obs_rms.normalize(x_np), dtype=torch.float32, device=self.device)
                    pred, tgt = self.rnd(x)
                    rnd_loss = F.mse_loss(pred, tgt.detach())
                    self.rnd_opt.zero_grad()
                    rnd_loss.backward()
                    self.rnd_opt.step()
                    rnd_losses.append(rnd_loss.item())
                    with torch.no_grad():
                        rnd_pred_errs.append(F.mse_loss(pred, tgt, reduction='none').mean().item())
            # Optional extra predictor-only epochs per PPO update
            for _ in range(self.rnd_update_epochs - 1):
                if not self.use_rnd:
                    break
                indices = torch.randperm(dataset_size, device=self.device)
                for start in range(0, dataset_size, minibatch_size):
                    end = start + minibatch_size
                    mb = indices[start:end]
                    x = next_obs[mb]
                    if self.rnd_obs_norm:
                        x_np = x.detach().cpu().numpy()
                        self.obs_rms.update(x_np)
                        x = torch.as_tensor(self.obs_rms.normalize(x_np), dtype=torch.float32, device=self.device)
                    pred, tgt = self.rnd(x)
                    rnd_loss = F.mse_loss(pred, tgt.detach())
                    self.rnd_opt.zero_grad()
                    rnd_loss.backward()
                    self.rnd_opt.step()
                    rnd_losses.append(rnd_loss.item())
                
        # Log training metrics
        training_metrics = {
            'train/policy_loss': np.mean(policy_losses),
            'train/value_loss': np.mean(value_losses),
            'train/entropy_loss': np.mean(entropy_losses),
            'train/total_loss': np.mean(total_losses),
            'train/kl_divergence': np.mean(kl_divergences),
            'train/clip_fraction': np.mean(clip_fractions),
            'train/learning_rate': self.optimizer.param_groups[0]['lr']
        }

        if self.use_rnd and rnd_losses:
            training_metrics.update({
                'train/rnd_loss': np.mean(rnd_losses),
                'train/rnd_pred_err': np.mean(rnd_pred_errs)
            })
        self.logger.log_metrics(training_metrics, self.total_timesteps)

        return {
            'policy_loss': np.mean(policy_losses),
            'total_loss': np.mean(total_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences)
        }
    
    def train(self, total_timesteps=1000000, rollout_size=2048, log_interval=10, sil=False):
        '''Main training loop'''
        
        print('Starting PPO training...')
        print(f'Device: {self.device}')
        print(f'Observation dim: {self.obs_dim}, Action dim: {self.action_dim}')
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
            avg_ext = np.mean(self.episode_ext_rewards) if self.episode_ext_rewards else 0.0
            avg_int = np.mean(self.episode_int_rewards) if (self.use_rnd and self.episode_int_rewards) else 0.0
            avg_comb = np.mean(self.episode_comb_rewards) if self.episode_comb_rewards else avg_ext
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            success_rate = np.mean(self.episode_successes) if self.episode_successes else 0
            
            # Calculate training speed
            elapsed_time = time.time() - self.start_time
            fps = self.total_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            # Log wandb metrics
            general_metrics = {
                'general/timesteps': self.total_timesteps,
                'general/updates': self.update_count,
                'general/fps': fps,
                'general/avg_episode_reward': float(avg_ext),
                'general/avg_intrinsic_reward': float(avg_int),
                'general/avg_combined_reward': float(avg_comb),
                'general/avg_episode_length': float(avg_length),
                'general/success_rate': float(success_rate),
                'general/num_episodes': len(self.episode_rewards),
            }
            self.logger.log_metrics(general_metrics, self.total_timesteps)
            
            # Store learning curve data  (For sample efficency)
            self.learning_curve.append({
                'timesteps': self.total_timesteps,
                'avg_reward': float(avg_ext)  # Use combined reward for learning curve
            })
            
            # Store additional curves
            self.episode_length_curve.append({
                'timesteps': self.total_timesteps,
                'avg_episode_length': avg_length
            })
            
            self.success_rate_curve.append({
                'timesteps': self.total_timesteps,
                'success_rate': success_rate
            })
            
            self.policy_behavior_curve.append({
                'timesteps': self.total_timesteps,
                'policy_loss': policy_metrics['policy_loss'],
                'total_loss': policy_metrics['total_loss'],
                'entropy_loss': policy_metrics['entropy_loss'],
                'kl_divergence': policy_metrics['kl_divergence']
            })
            
            


            print(f'Update {self.update_count}')
            print(f'Timesteps: {self.total_timesteps}/{total_timesteps}')
            print(f'FPS: {fps:.2f}')    
            print(f'Avg Episode Reward: {avg_ext:.2f}')
            print(f'Avg Intrinsic Reward: {avg_int:.2f}' if self.use_rnd else '')
            print(f'Avg Episode Length: {avg_length:.2f}')
            print(f'Success Rate: {success_rate:.2f}')
            print('-' * 50)
        
        # Close logging
        
        print('Training completed')

    def get_learning_curve(self):
        return self.learning_curve
    
    def get_episode_length_curve(self):
        return self.episode_length_curve
    
    def get_success_rate_curve(self):
        return self.success_rate_curve
    
    def get_policy_behavior_curve(self):
        return self.policy_behavior_curve
    
    def evaluate(self, num_episodes=10, render=False, log_video=False):
        '''Evaluate trained policy'''
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
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
                        # frame = self.env.render(mode='rgb_array')
                        frame = self.env.render()
                        if frame is not None:
                            frames.append(frame)
                
                with torch.no_grad():
                    action, _, _, _ = self.network.get_action_and_value(obs)
                
                obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
                done = terminated or truncated
                if done:
                    success = terminated 
                    eval_successes.append(success)
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
            'eval/max_reward': np.max(eval_rewards),
            'eval/min_reward': np.min(eval_rewards),
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