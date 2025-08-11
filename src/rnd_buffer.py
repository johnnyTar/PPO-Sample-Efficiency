import torch


class PPORNDBuffer:
    '''
    Experience buffer for PPO with Generalized Advantage Estimation (GAE)

    Implement an experience buffer PPO that stores trajectory data and computes advantages using Generalized Advantage
    Estimation (GAE). The buffer handles multiple episodes within a single
    rollout and provides normalized advantages for stable policy updates.

    Key Features:
    1. Storage: Collect transitions during environment interaction
    2. Processing: Compute advantages and returns using GAE
    3. Retrieval: Return normalized batch for policy updates
    '''
    
    def __init__(self, size, obs_dim, device, gamma=0.99, lam=0.95):
        '''
        size: number of timesteps
        obs_dim: Dimensionality of observations (flattened NN)
        gamma: Discount factor to future rewards
        lam: GAE for bias-variance tradeoff
                    - Lower values reduce variance but increase bias
                    - Higher values reduce bias but increase variance
        '''
        self.size = size
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.reset()
        
    def reset(self):
        """
        Reset buffer storage
        This method initializes the buffer to store trajectory data.
        """
        self.observations = torch.zeros((self.size, self.obs_dim), device=self.device)
        self.actions = torch.zeros(self.size, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(self.size, device=self.device)
        self.values = torch.zeros(self.size, device=self.device)
        self.log_probs = torch.zeros(self.size, device=self.device)
        self.dones = torch.zeros(self.size, device=self.device)
        self.advantages = torch.zeros(self.size, device=self.device)
        self.returns = torch.zeros(self.size, device=self.device)
        self.ptr = 0
        self.path_start_idx = 0
        self.next_observations = torch.zeros((self.size, self.obs_dim), device=self.device)
        
    def store(self, obs, act, rew, val, log_prob, done, next_obs):
        ''''Store a single transition in the buffer.'''
        assert self.ptr < self.size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.next_observations[self.ptr] = next_obs
        self.ptr += 1
        
    def finish_path(self, last_val=0):
        '''
        This method implements the Generalized Advantage Estimation algorithm
        to compute advantage estimates for the current trajectory
        GAE to balance bias and variance in advantage estimation
        by combining n.step returns
        
        GAE Formula:
        A_t^GAE = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
        '''
        # Get the slice of the current path
        # This is the range of indices for the current episode
        # It allows us to compute advantages only for the current episode
        path_slice = slice(self.path_start_idx, self.ptr)
        path_len = self.ptr - self.path_start_idx
        
        if path_len == 0:
            return
            
        # Get rewards, values, and dones for this path
        path_rewards = self.rewards[path_slice]
        path_values = self.values[path_slice]
        path_dones = self.dones[path_slice]
        
        # GAE-Lambda advantage estimation
        advantages = torch.zeros(path_len, device=self.device)
        gae = 0
        
        for t in reversed(range(path_len)):
            if t == path_len - 1:
                next_non_terminal = 1.0 - path_dones[t]
                next_values = last_val
            else:
                next_non_terminal = 1.0 - path_dones[t]
                next_values = path_values[t + 1]
            # TD-error
            delta = path_rewards[t] + self.gamma * next_values * next_non_terminal - path_values[t]
            # GAE calc
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae
            
        # Returns = advantages + values
        returns = advantages + path_values
        
        # Store in buffer
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
        
    def get_batch(self):
        '''
        Retrieve batch with normalized advantages for policy updates
        Advantages are normalized (zero mean , unit variance) to improve training
        stability and convergence speed.
        '''
        assert self.ptr == self.size
        self.ptr = 0
        self.path_start_idx = 0
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        advantages_normalized = (self.advantages - adv_mean) / adv_std
        
        batch = {
            'obs': self.observations.clone(),
            'act': self.actions.clone(),
            'ret': self.returns.clone(),
            'adv': advantages_normalized,
            'logp': self.log_probs.clone(),
            'next_obs': self.next_observations.clone()
        }
        
        # Reset for next rollout
        self.reset()
        
        return batch