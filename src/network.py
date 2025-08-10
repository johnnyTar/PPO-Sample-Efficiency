import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    '''
    Neural network for PPO with shared layers and separate heads for actor/critic
    This design allow:
    the policy and value function to share learned representations while maintaining separate
    outputs for their distinct objectives. and Computational efficiency

    Key Features:
    - Shared feature extraction layers
    - Separate actor head for policy (action probability distribution)
    - Separate critic head for value function (state value estimation)
    - Categorical action distribution for discrete action spaces
    - Unified interface for action sampling and evaluation

    Mathematical Framework:
    - Actor output: p(a|s) probability distribution over actions given state
    - Critic output: V(s) expected return from state s
    - Action sampling: a Categorical(logits) for discrete actions
    - Policy evaluation: log p(a|s) for given state-action pairs
    '''
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        '''
        The network uses a simple feedforward design with ReLU activations func.
        - Input layer: obs_dim -> hidden_dim
        - Hidden layers: hidden_dim -> hidden_dim (2 layers)
        - Actor head: hidden_dim -> action_dim (policy logits)
        - Critic head: hidden_dim -> 1 (value estimate)
        '''
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
        '''
        obs_dim -> hidden_dim() -> hidden_dim -> action_logits (6) -> softmax-> action probs dist (6)
        '''
        shared_features = self.shared(x)
        # Convert features to action logits
        action_logits = self.actor(shared_features)
        # Get expected return from givin state
        value = self.critic(shared_features) # 1 dim output
        return action_logits, value
    
    def get_action_and_value(self, x, action=None):
        '''
        Get action and value with probability distribution.
        
        1. Sample actions from the policy
        2. Get log probabilities of given actions
        '''
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value