import os
import wandb
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class PPOLogger:
    '''
    Handles logging for PPO training with WandB and TensorBoard support
    '''
    
    def __init__(self, experiment_name=None, use_wandb=True, use_tensorboard=True, 
                env=None, network=None, config=None):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        self.experiment_name = experiment_name
        
        # Store configation
        self.config = config if config is not None else {}
        
        # Setup logging
        self._setup_wandb(env, network, config)
        self._setup_tensorboard(config)
        
    def _setup_wandb(self, env, network, config):
        '''Setup WandB logging'''
        if self.use_wandb:
            wandb_config = {
                'algorithm': 'PPO',
                'environment': str(env.spec.id) if hasattr(env, 'spec') else 'MiniGrid',
                'device': str(config.get('device', 'cpu'))
            }
            
            # Add config parameters if provided
            if config:
                wandb_config.update(config)
                
            wandb.init(
                project='ppo-minigrid',# Project name
                name=self.experiment_name,
                config=wandb_config
            )
            
            # Watch the model if provided
            if network is not None:
                wandb.watch(network, log='all', log_freq=100)
    
    def _setup_tensorboard(self, config):
        '''Setup TensorBoard logging'''
        if self.use_tensorboard:
            log_dir = f'runs/{self.experiment_name}'
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            
            # Log hyperparameters if provided
            if config:
                hparams = {k: v for k, v in config.items() 
                            if isinstance(v, (int, float, str, bool))}
                self.writer.add_hparams(hparams, {})
    
    def log_metrics(self, metrics_dict, step=None):
        '''Log metrics to WandB and TensorBoard'''
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)
        
        if self.use_tensorboard:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step or 0)
    
    def log_video(self, frames, key='eval/video', fps=4):
        '''Log video to WandB'''
        # TODO: Log video not working for Minigrid 
        if self.use_wandb and len(frames) > 0:
            import numpy as np
            wandb.log({key: wandb.Video(np.array(frames), fps=fps, format='gif')})
    

    
    def close(self):
        '''Close logging connections'''
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.writer.close()
    
    def get_wandb_url(self):
        '''Get WandB run URL'''
        if self.use_wandb and wandb.run:
            return f'https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}'
        return None