import os
import argparse
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from gymnasium.wrappers import RecordVideo, TimeLimit
from datetime import datetime
import time

# Import both agent types from the new separate files
from agent import PPOAgent
from sil_agent import PPOSILAgent
from utils import set_seed
import numpy as np
import json

def parse_args():
    
    '''
    Arguments for PPO training on MiniGrid both standard PPO and 
    PPO with Self-Imitation Learning (SIL) on MiniGrid environments.

    The defaults correspond to the configuration reported in 
    "Hyperparameters in RL and How to Tune Them" (Eimer et al., 2023) for MiniGrid-Empty-5x5 and DoorKey-5x5.
    '''
    parser = argparse.ArgumentParser(description='PPO Training Script')
    
    # =================================================================
    # Experiment Configuration
    # =================================================================
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'),
                        help='Name of the experiment')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='Seed of the experiment for reproducibility')
    
    # =================================================================
    # Environment Configuration
    # =================================================================
    parser.add_argument('--gym-id', type=str, default='MiniGrid-DoorKey-6x6-v0',
                        help='ID of the Gym environment to use')
    
    # =================================================================
    # Core Training Hyperparameters
    # =================================================================
    parser.add_argument('--total-timesteps', type=int, default=1_000_000, # 500_000 1_000_000
                        help='Total number of timesteps to train the agent')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--rollout-size', type=int, default=1024,
                        help='Number of steps to collect per rollout')
    
    # =================================================================
    # PPO Algorithm Parameters
    # =================================================================
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for rewards') # Discount factor for rewards
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='Lambda for GAE advantage estimation') # Lambda for GAE, controls the bias-variance tradeoff
    parser.add_argument('--clip-coef', type=float, default=0.14,
                        help='Clipping coefficient for PPO') # Clipping coefficient for PPO, controls the policy
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Coefficient for the entropy bonus') # Coefficient for the entropy bonus, encourages exploration
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='Coefficient for the value function loss') # Coefficient for the value function loss, controls the importance of the value function
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Mini-batch size for optimisation')
    
    # =================================================================
    # Self-Imitation Learning Configuration
    # =================================================================
    parser.add_argument('--sil', type=lambda x: x.lower() == 'true', default=False,
                        help='Enable/disable Self-Imitation Learning (true/false)')
    # SIL Algorithm Parameters
    parser.add_argument('--sil-lr', type=float, default=1e-4,
                        help='Learning rate for SIL updates')
    parser.add_argument('--sil-buffer-size', type=int, default=10000,
                        help='Size of SIL replay buffer')
    parser.add_argument('--sil-batch-size', type=int, default=32,
                        help='Batch size for SIL updates')
    parser.add_argument('--sil-update-ratio', type=int, default=4,
                        help='Number of SIL updates per PPO update')
    # SIL Prioritized Experience Replay Parameters
    parser.add_argument('--sil-alpha', type=float, default=0.6,
                        help='Prioritization exponent for SIL buffer')
    parser.add_argument('--sil-beta', type=float, default=0.4,
                        help='Importance sampling exponent for SIL')
    parser.add_argument('--sil-clip-coef', type=float, default=None,
                        help='Clipping coefficient for SIL (default: use same as PPO)')
    parser.add_argument('--success-threshold', type=float, default=0.8,
                        help='Success threshold for SIL buffer')
    parser.add_argument('--sil-warmup-episodes', type=int, default=50,
                        help='Number of episodes before SIL updates start')
    parser.add_argument('--sil-update-frequency', type=int, default=5,
                        help='SIL update frequency (every N PPO updates)')
    
    # =================================================================
    # Logging and Monitoring
    # =================================================================
    parser.add_argument('--track', action='store_true', default=False,
                        help='Use wandb to track the experiment')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Disable tensorboard logging')
    
    return parser.parse_args()


def train_single_seed(args, seed):
    '''
    Train a agent with the specified seed.
    
    1. Environment setup with seeding 
    2. Agent creation (PPO or PPO-SIL)
    3. Training execution with specified hyperparameters
    4. Evaluation of trained agent
    5. Model saving and result collection
    '''
    
    # Set global random seed for reproducibility
    set_seed(seed)
    
    # Create MiniGrid environment
    env = gym.make(args.gym_id, render_mode='rgb_array')
    env = FlatObsWrapper(env)  # Converts dict observations to flat vectors for neural networks-> Flatten observation for easier handling
    env = TimeLimit(env, max_episode_steps=1000) # Original for cartpole
    
    # Seed environment
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Setup video recording with seed in name
    sil_add = "_SIL" if args.sil else "_PPO"
    run_name = f"{args.gym_id}_seed_{seed}{sil_add}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    video_dir = os.path.join('videos', run_name)
    os.makedirs(video_dir, exist_ok=True)
    
    env = RecordVideo(
        env,
        f'videos/{run_name}',
        episode_trigger=lambda ep: ep % 250 == 0,  # Record every 250 episodes
        video_length=200
    )
    
    # Create agent
    if args.sil:
        print(f"PPO+SIL {seed}")
        agent = PPOSILAgent(
            env, 
            hidden_dim=256, 
            seed=seed,
            lr=args.learning_rate,
            gamma=args.gamma,
            lam=args.gae_lambda,
            clip_ratio=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            batch_size=args.batch_size,
            # SIL parameters
            sil=True,
            sil_lr=args.sil_lr,
            sil_buffer_size=args.sil_buffer_size,
            sil_batch_size=args.sil_batch_size,
            sil_update_ratio=args.sil_update_ratio,
            sil_alpha=args.sil_alpha,
            sil_beta=args.sil_beta,
            sil_clip_ratio=args.sil_clip_coef,
            success_threshold=args.success_threshold,
            sil_warmup_episodes=args.sil_warmup_episodes,
            sil_update_frequency=args.sil_update_frequency,
            # Logging
            use_wandb=args.track,
            use_tensorboard=args.tensorboard,
            experiment_name=run_name
        )
    else:
        print(f"PPO {seed}")
        agent = PPOAgent(
            env, 
            hidden_dim=256, 
            seed=seed,
            lr=args.learning_rate,
            gamma=args.gamma,
            lam=args.gae_lambda,
            clip_ratio=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            batch_size=args.batch_size,
            # Logging
            use_wandb=args.track,
            use_tensorboard=args.tensorboard,
            experiment_name=run_name
        )
    
    # Train the agent
    agent.train(
        total_timesteps=args.total_timesteps, 
        rollout_size=args.rollout_size, 
        sil=args.sil,
    )
    
    # Evaluate the trained agent
    print(f'\nEvaluating trained agent (seed {seed})')
    mean_reward, std_reward = agent.evaluate(num_episodes=10, log_video=False)
    print(f'Mean evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}')
    
    # Save the trained model with seed in filename
    model_dir = os.path.join('models', run_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = agent.save_model(os.path.join(model_dir, f'{run_name}.pth'))

    print(f'Training completed')
    
    if agent.logger.use_wandb:
        wandb_url = agent.logger.get_wandb_url()
        if wandb_url:
            print(f'- WandB: {wandb_url}')
    
    if agent.logger.use_tensorboard:
        print(f'TensorBoard: {agent.logger.experiment_name}')

    return {
        'seed': seed,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'learning_curve': agent.get_learning_curve(),
        'success_rate_curve': agent.get_success_rate_curve(),
        'episode_length_curve': agent.get_episode_length_curve(),
        'policy_behavior_curve': agent.get_policy_behavior_curve(),
        'sil': args.sil,
    }


def main(args):
    '''Main training script with multiple seeds'''
    
    results = []
    
    # Train agent for each seed
    for seed in args.seeds:
        print(f'\n{"="*60}')
        print(f'Training with seed: {seed}')
        print(f'{"="*60}')
        
        result = train_single_seed(args, seed)
        results.append(result)
    
    # Save and print summary results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    sil_add = "_SIL" if args.sil else "_PPO"
    results_file = os.path.join(results_dir, f'{args.gym_id}{sil_add}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n{"="*60}')
    print('SUMARY RESULTS')
    print(f'{"="*60}')
    all_means = [r['mean_reward'] for r in results]
    print(f'Overall Mean +- Std: {np.mean(all_means):.2f} +- {np.std(all_means):.2f}')
    print(f'Results saved to: {results_file}')
    
    # Display individual seed results
    for result in results:
        print(f"Seed {result['seed']}: {result['mean_reward']:.2f} +- {result['std_reward']:.2f}")
    
    return results


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    print(f'\n{"="*60}')
    print(f'Training with configs:')
    print(f'{"="*60}')
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')
    print(f'{"="*60}')
    
    # Train with parsed arguments
    # mean_reward, std_reward = main(args)
    results = main(args)
    
    print(f'\n{"="*60}')
    print('TRAINING COMPLETED')
    print(f'{"="*60}')