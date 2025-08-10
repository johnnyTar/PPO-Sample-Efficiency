import torch
import numpy as np
from collections import deque


class SILBuffer:
    """
    Self-Imitation Learning Buffer: Agents often achieve good 
    performance by chance but struggle to reproduce it. 
    SIL solves this by storing and learning from past successes.
    
    This buffer implements a experience replay that:
    1. Stores only successful episodes (self-imitation)
    2. Uses prioritized sampling to focus on best experiences
    3. Adapts its success criteria as the agent improves
    4. Integrates seamlessly with PPO-style algorithms
    
    Similar to:
    - "Self-Imitation Learning" (Oh et al., 2018)
    - "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    def __init__(self, capacity=10000, obs_dim=None, device='cpu', alpha=0.6, beta=0.4, 
                 beta_increment=1e-6, success_threshold=0.8, min_episode_length=10):
        self.capacity = capacity # Maximum number of transitions to store
        self.obs_dim = obs_dim
        self.device = device
        self.alpha = alpha  # Prioritization exponent [0, 1], 0 Uniform sampling (no prio), 1 Full prioritization (only sample best)
        self.beta = beta # Importance sampling correction [0, 1] 
        self.beta_increment = beta_increment  # For beta -> Starts low (0.4) for aggressive prioritization -> Anneals to 1.0 for unbiased learning
        self.epsilon = 1e-6 # Constant to ensure non-zero priorities
        
        # Filtering parameters
        self.success_threshold = success_threshold # what episodes to keep
        self.min_episode_length = min_episode_length
        self.use_success_only = True # Opnly store successes
        self.adaptive_threshold = True
        
        # Pre-allocated tensors (PPO style)
        self.observations = None
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.returns = torch.zeros(capacity, device=device) # Return-to-go val
        self.log_probs = torch.zeros(capacity, device=device) # p_old(a|s) for PPO
        self.values = torch.zeros(capacity, device=device)# V_old(s) for advantag
        self.episode_ids = torch.zeros(capacity, dtype=torch.long, device=device)
        
        # Priority tracking: transitions with higher returns are sampled more freq
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Episode tracking
        self.episode_buffer = [] # Temporare store for ongoing episode
        self.episode_count = 0 # Total episodes seen
        self.episode_returns = deque(maxlen=200)
        self.total_episodes_added = 0
        self.successful_episodes = deque(maxlen=100)
        self.threshold_history = deque(maxlen=50)
        
    def _init_obs_buffer(self, obs_shape):
        '''
        Initialize observation buffer based on first observation
        Used for RND
        '''
        if len(obs_shape) == 1:
            self.observations = torch.zeros((self.capacity, obs_shape[0]), device=self.device)
        else:
            self.observations = torch.zeros((self.capacity, *obs_shape), device=self.device)
    
    def start_episode(self):
        """Begin collecting a new episode."""
        self.episode_buffer = []
        self.episode_count += 1
        
    def add_transition(self, obs, action, reward, log_prob, value):
        """Add a transition to the current episode buffer"""
        obs = obs.to(self.device) # Current observation
        action = action.to(self.device) # 
        log_prob = log_prob.to(self.device) # Log probability of action under current policy
        value = value.to(self.device) # Value estimate V(s) from critic
        
        transition = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value
        }
        self.episode_buffer.append(transition)
        
    def _compute_rewards(self, episode_buffer, final_reward):
        """
        Apply reward to encourage efficient solutions.
        
        STRATEGY:
        1. Progress bonus: Decreases over episode (rewards quick solutions)
        2. Step bonus: Small constant for surviving (encourages exploration)
        3. The 10-step solution gets higher cumulative rewards due to larger progress bonuses
            This makes shorter, more efficient solutions more attractive during learning
            The agent learns to prefer quick wins over slow wins
        """
        new_rewards = []
        episode_length = len(episode_buffer)
        
        for i, transition in enumerate(episode_buffer):
            base_reward = transition['reward']
            
            # Length-based rewward for successful episodes
            if final_reward > 0:  # Successful episode
                # Progress bonus: linearly decreases from 0.1 to 0
                progress_bonus = (episode_length - i) / episode_length * 0.1
                # Small reward for each step in successful episode
                # Differentiates between trajectories of same final reward
                new_reward = base_reward + progress_bonus + 0.01
            else:
                print('For failed episodes, dont add positive')
                # For failed episodes, dont add positive 
                new_reward = base_reward
            
            new_rewards.append(new_reward)
        
        return new_rewards
        
    def _should_store_episode(self, episode_return, episode_length):
        """
        Decide if an episode should be stored in buffer
        
        FILTERING:
        1. Length filter: Reject trivially short episodes (not used for minigrid)
        2. Success filter: Positive return 
        3. Quality filter: Must exceed current performance threshold
        
        ADAPTIVE:
        - Threshold rises as agent improves (maintain high standards)
        - Threshold lowers if too few successes (prevent deadlock)
        
        """
        
        # Success-only filter
        if self.use_success_only and episode_return <= 0:
            return False
        
        # Adaptive threshold
        current_threshold = self.success_threshold
        # Aadjust threshold based on recent performance
        if self.adaptive_threshold and len(self.episode_returns) > 20:
            recent_mean = np.mean(list(self.episode_returns)[-20:]) # Average of last 20
            recent_max = np.max(list(self.episode_returns)[-50:]) # Best of last 50
            
            # Adapt threshold based on recent performance
            if recent_max > current_threshold:
                current_threshold = max(current_threshold, recent_mean + 0.1)
            elif len(self.successful_episodes) < 5:  # Not enough good episodes
                current_threshold = max(0.5, current_threshold * 0.9)  # Lower threshold
        
        # Check against threshold
        if episode_return < current_threshold:
            return False
        
        # Store threshold for tracking
        self.threshold_history.append(current_threshold)
        
        return True
        
    def finish_episode(self, episode_return, gamma=0.99):
        """
        Process completed episode and store if it meets quality
        1. Evaluate episode quality
        2. Compute returns-to-go
        3. Apply reward
        4. Store in buffer if good enough
        """
        if len(self.episode_buffer) == 0:
            return
        
        self.episode_returns.append(episode_return)
        episode_length = len(self.episode_buffer)
        
        # Check if episode should be stored
        if not self._should_store_episode(episode_return, episode_length):
            self.episode_buffer = []
            return
        
        # Episode passed filters - increment counter and track success
        self.total_episodes_added += 1
        self.successful_episodes.append(episode_return)
        
        # Initialize observation buffer on first episode
        if self.observations is None:
            first_obs = self.episode_buffer[0]['obs']
            self._init_obs_buffer(first_obs.shape)
        
        # Compute rewards
        shaped_rewards = self._compute_rewards(self.episode_buffer, episode_return)
        
        # Compute cumulative discounted rewards 
        returns_to_go = []
        running_return = 0
        
        for i in reversed(range(len(self.episode_buffer))):
            running_return = shaped_rewards[i] + gamma * running_return
            returns_to_go.insert(0, running_return)
        
        # Add transitions to buffer (PPO style)
        for transition, return_val in zip(self.episode_buffer, returns_to_go):
            self._store_transition(
                transition['obs'],
                transition['action'],
                return_val,
                transition['log_prob'],
                transition['value']
            )
        
        self.episode_buffer = []
        print(f"Added episode to SIL buffer: reward={episode_return:.2f}, "
              f"length={episode_length}, buffer_size={self.size}")
        
    def _store_transition(self, obs, action, return_val, log_prob, value):
        """
        Store a single transition in the main buffer.
        
        STORAGE STRATEGY:
        1. Overwrit oldest when full
        2. Priority-based sampling preparation
        """
        # Calculate priority for sampling
        # Higher returns = higher priority = sampled more 
        # Rpsilon: Ensure non-zero priority 
        priority = max(return_val, self.epsilon)
        
        # Store in pre-allocated tensors
        self.observations[self.position] = obs
        self.actions[self.position] = action.squeeze() if action.dim() > 0 else action
        self.returns[self.position] = return_val
        self.log_probs[self.position] = log_prob.squeeze() if log_prob.dim() > 0 else log_prob
        self.values[self.position] = value.squeeze() if value.dim() > 0 else value
        self.episode_ids[self.position] = self.episode_count
        
        # Set priority for this transition
        # alpha: Control how much to prioritize (0=uniform, 1=greedy))
        self.priorities[self.position] = priority ** self.alpha
        
        # Update pointers
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch using prioritized experience replay.
        
        SAMPLING STRATEGY:
        1. Higher return transitions sampled more frequently
        2. Importance sampling weights correct for bias
        3. Beta annealing: removes bias over training
        """
        if self.size == 0:
            return None
            
        # Calc sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / np.sum(priorities)
        
        # Sample according to priorities
        indices = np.random.choice(self.size, size=min(batch_size, self.size), 
                                 p=probabilities, replace=True)
        
        # Importance sampling weight calculation to corrects the bias
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights) # Normalize weights
        
        # Get batch data (PPO style)
        batch = {
            'obs': self.observations[indices].clone(),
            'act': self.actions[indices].clone(),
            'ret': self.returns[indices].clone(),
            'logp': self.log_probs[indices].clone(),
            'val': self.values[indices].clone(),
            'weights': torch.FloatTensor(weights).to(self.device),
            'indices': indices
        }
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities after training from sampled transitions.
        
        Priorities change wehn value estimates improve.
        """
        for idx, priority in zip(indices, priorities):
            if idx < self.size:  # Ensure valid index
                self.priorities[idx] = (priority + self.epsilon) ** self.alpha
    
    def get_stats(self):
        if self.size == 0:
            return {
                'buffer_size': 0,
                'episodes_added': self.total_episodes_added,
                'beta': self.beta
            }
            
        return {
            'buffer_size': self.size,
            'episodes_added': self.total_episodes_added,
            'mean_return': self.returns[:self.size].mean().item(),
            'max_return': self.returns[:self.size].max().item(),
            'min_return': self.returns[:self.size].min().item(),
            'beta': self.beta,
            'current_threshold': self.threshold_history[-1] if self.threshold_history else self.success_threshold,
            'success_rate': len(self.successful_episodes) / max(len(self.episode_returns), 1)
        }
    def __len__(self):
        return self.size