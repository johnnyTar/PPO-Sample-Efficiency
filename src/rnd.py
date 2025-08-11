import torch 
import torch.nn as nn 
import numpy as np


class RunningMeanStd:
    """
    A class to maintain running mean and standard deviation for normalization.
    Useful for normalizing observations or rewards in reinforcement learning.
    """
    def __init__(self, shape=(), eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0] if x.ndim > 0 else 1.0
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x, eps=1e-8):
        return (x - self.mean) / np.sqrt(self.var + eps)
    


class RNDModel(nn.Module):
    """
    A simple Random Network Distillation (RND) model.
    It consists of a target network and a predictor network.
    The target network is fixed and used to compute intrinsic rewards,
    while the predictor network is trained to predict the target network's output.
    """

    def __init__(self, obs_dim, embed_dim=128, hidden_dim=256):
        """
        The network consists of two MLPs:
        args:
            obs_dim: Dimension of the input observation.
            embed_dim: Dimension of the output embedding.
            hidden_dim: Dimension of the hidden layers in the MLPs.
        """
        super().__init__()
        def mlp(in_dim, out_dim):
            return nn.Sequential( nn.Linear(in_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim)
                                 )
        
        self.target = mlp(obs_dim, embed_dim)
        self.predictor = mlp(obs_dim, embed_dim)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        """
        Forward pass through the model.
        Args:
            obs: Input observation tensor.
        Returns:
            p: Output of the predictor network.
            t: Output of the target network.
        """
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        return p, t