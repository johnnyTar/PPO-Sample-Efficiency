import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    '''Neural network for PPO with shared layers and separate heads for actor/critic'''
    
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