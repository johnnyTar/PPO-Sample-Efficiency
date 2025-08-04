import torch


class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, size, obs_dim, device, gamma=0.99, lam=0.95):
        self.size = size
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.reset()
        
    def reset(self):
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
                
            delta = path_rewards[t] + self.gamma * next_values * next_non_terminal - path_values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae
            
        # Returns = advantages + values
        returns = advantages + path_values
        
        # Store in buffer
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
        advantages_normalized = (self.advantages - adv_mean) / adv_std
        
        batch = {
            'obs': self.observations.clone(),
            'act': self.actions.clone(),
            'ret': self.returns.clone(),
            'adv': advantages_normalized,
            'logp': self.log_probs.clone()
        }
        
        # Reset for next rollout
        self.reset()
        
        return batch