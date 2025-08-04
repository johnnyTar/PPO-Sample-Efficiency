"""Utility functions for PPO training"""

import os
import torch
import numpy as np
import random
from datetime import datetime


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(base_dir="experiments"):
    """Create necessary directories for experiments"""
    directories = [
        base_dir,
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "videos"),
        os.path.join(base_dir, "logs"),
        "runs"  # For TensorBoard logs
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories


def generate_experiment_name(prefix="PPO_MiniGrid"):
    """Generate a unique experiment name with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}"


def load_model(model_path, network, optimizer=None, device=None):
    """Load a saved model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    config = checkpoint.get('config', {})
    print(f"Model loaded from {model_path}")
    print(f"Model config: {config}")
    
    return network, optimizer, config


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info():
    """Get information about available devices"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA device: {gpu_name} ({gpu_memory:.1f}GB)"
    else:
        return "CPU device"


def format_time(seconds):
    """Format time in seconds to human readable format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def normalize_advantages(advantages):
    """Normalize advantages to have zero mean and unit variance"""
    advantages = np.array(advantages)
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)