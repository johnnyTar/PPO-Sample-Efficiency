

import argparse
import os
import random
import time
from distutils.util import strtobool
import numpy as np
import torch
import gym
import glob
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training Script')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rsplit('.py')[0],
                        help='Name of the experiment') # Ecperiment name, value of the file name without extension
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
                        help='ID of the Gym environment to use')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed of the experiment for reproducibility')
    parser.add_argument('--total-timesteps', type=int, default=10000,
                        help='Total number of timesteps to train the agent')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`') # Reproducibility setting for PyTorch
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will be enabled by default') # Enable GPU support if available    
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, use wandb to track the experiment') # Use Weights & Biases for tracking
    parser.add_argument('--wandb-project-name', type=str, default='ppo-sample-efficient',
                        help="the wandb's project name") # Project name for Weights & Biases
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project") # Entity name for Weights & Biases, can be None = you name
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)') # Capture videos of the agents performance
    
    parser.add_argument('--num-envs', type=int, default=4,  
                        help='number of parallel enviroments') # Number of parallel environments to run, 
    parser.add_argument('--num-steps', type=int, default=128,
                        help='number of steps to run in each environment per update') # Contols how much data is collected before updating the agent
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, the learning rate will be annealed to zero') # Anneal the learning rate to zero over the course of training
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, use Generalized Advantage Estimation (GAE)') # Use GAE for advantage estimation
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards') # Discount factor for rewards
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='lambda for GAE') # Lambda for GAE, controls the bias-variance tradeoff
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='number of mini-batches to split the data into for each update') # Number of mini-batches to split the data into for each update
    parser.add_argument('--num-epochs', type=int, default=4,
                        help='number of epochs to update policy') # Number of epochs to update the agent for each update
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, normalize the advantages') # Normalize the advantages to have zero mean and unit variance  
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help='clipping coefficient for PPO') # Clipping coefficient for PPO, controls the policy
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, clip the value loss') # Clip the value loss to prevent large updates
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='coefficient for the entropy bonus') # Coefficient for the entropy bonus, encourages exploration
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='coefficient for the value function loss') # Coefficient for the value function loss, controls the importance of the value function
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='maximum norm for the gradients') # Maximum norm for the gradients, prevents exploding gradients
    parser.add_argument('--target-kl', type=float, default=0.015,
                        help='target KL divergence for early stopping') # Target KL divergence for early stopping, prevents
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)  # Batch size for training 
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # Number of updates to perform
    args.num_updates = args.total_timesteps // args.batch_size # Number of updates to perform

    return args


def log_videos_to_wandb(video_dir, step=None):
    """Log all videos in the video directory to wandb"""
    if not os.path.exists(video_dir):
        return
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    for video_file in video_files:
        try:
            wandb.log({
                "video": wandb.Video(video_file, format="mp4")
            }, step=step)
            # print(f"Logged video to wandb: {video_file}")
        except Exception as e:
            print(f"Failed to log video {video_file}: {e}")

# PPO deals with vector environments, so we need to wrap the environment
def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():        
        if capture_video and idx == 0:  # Only capture video for the first environment
            os.makedirs(f'videos/{run_name}', exist_ok=True)
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f'videos/{run_name}',
                episode_trigger=lambda ep: True,
                video_length=200,
            )
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with a given standard deviation and bias constant."""
    torch.nn.init.orthogonal_(layer.weight, std) # PPO uses orthogonal initialization on the layers weights
    # if layer.bias is not None:
    torch.nn.init.constant_(layer.bias, bias_const) # constant initialization on the layers bias
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential( # 3 linear layers with Tanh activation
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0, bias_const=0.0)
        )

        self.actor = nn.Sequential( # 3 linear layers with Tanh activation
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01, bias_const=0.0)
        )
    # Inference method for the agent
    def get_value(self, x):
        """Get the value of the state."""
        return self.critic(x)
    def get_action_and_value(self, x, action=None):
        """Get the action and value of the state."""
        logits = self.actor(x) # Unnormalized logits for the action distribution
        dist = Categorical(logits=logits) # Softmax distribution over the actions

        if action is None:  # If no action is provided, sample from the distribution
            action = dist.sample()  # Sample an action from the distribution
        
        return action, dist.log_prob(action), dist.entropy(), self.get_value(x)


if __name__ == '__main__':
    args = parse_args()

    # Unique run name based on the experiment parameters
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    if args.track:        
        # Initialize Weights & Biases
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity, 
            sync_tensorboard=True, # Sync TensorBoard logs
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        # Log the hyperparameters to Weights & Biases
        wandb.config.update(vars(args))

    # Save the metrics to a folder named after the run
    writer = SummaryWriter(log_dir=os.path.join('runs', run_name))

    # Encode args varables as text
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|\n|-|-|\n%s' % ('\n'.join([f'|{k}|{v}|' for k, v in vars(args).items()])),
    )


    random.seed(args.seed) # Set random seed for reproducibility
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    if args.track and args.capture_video:
        video_dir = os.path.join('videos', run_name)
        os.makedirs(video_dir, exist_ok=True)
    
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )  # Create a vectorized environment for PPO

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "Only supports discrete action spaces for now"
    # print(f"Using environment: {args.gym_id} with {args.num_envs} parallel environments")
    # print(f"Environment action space: {envs.single_action_space.n}")
    # print(f"Environment observation space: {envs.single_observation_space.shape}")

    agent = Agent(envs).to(device)  # Initialize the agent
    # print(agent)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Optimizer for the agent, epsilon same as paper

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) # Collect observations + how many env
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device) # Reset the environment and get the initial observation
    next_done  = torch.zeros(args.num_envs).to(device) # Initialize dones
    # print(f"Total updates: {args.num_updates}, Batch size: {args.batch_size}")
    # print(f"next_obs shape: {next_obs.shape}, next_done shape: {next_done.shape}")
    # print(f"agent.get_value(next_obs): {agent.get_value(next_obs)}")
    # print(f"agent.get_value(next_obs).shape: {agent.get_value(next_obs).shape}")
    # print(f"\nagent.get_value(next_obs).squeeze().shape: {agent.get_value(next_obs).squeeze().shape}")
    # print(f"\nagent.get_action_and_value(next_obs) {agent.get_action_and_value(next_obs)}")

    for update in range(args.num_updates):
        # Anneal the learning rate if specified
        if args.anneal_lr:
            frac = 1.0 - (update / args.num_updates) # Linear annealing lr to zero
            lr = args.learning_rate * frac # Annealed learning rate
            optimizer.param_groups[0]['lr'] = lr # Update the learning rate in the optimizer

        for step in range(0, args.num_steps): # Collect data for each step Rollout
            global_step += args.num_envs  # Increment global step by the number of environments
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs) # Get the value of the state
                values[step] = value.squeeze()  # Store the value of the state
            
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, truncated, infos = envs.step(action.cpu().numpy())  # Step the environment
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # Store the reward
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.tensor(done | truncated).to(device)  # Update next_obs and next_done
            
            if 'final_info' in infos:
                for finfo in infos['final_info']:
                    if finfo is not None and 'episode' in finfo:
                        r = finfo['episode']['r']
                        print(f"global_step: {global_step}, episode reward: {r}")
                        
                        writer.add_scalar('charts/episodic_return', r, global_step)  # Log the episodic return
                        writer.add_scalar('charts/episodic_length', finfo['episode']['l'], global_step)  # Log the episodic length

                        #if args.track and args.capture_video:
                        #    log_videos_to_wandb(video_dir, global_step)

    
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1].float()
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # Reshape observations for batching
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1) 
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)

        # Optimization loop
        b_inds = np.arange(args.batch_size)  # Create indices for batching
        clipfracs = []  # List to store the clipping fractions

        for epoch in range(args.num_epochs):
            np.random.shuffle(b_inds) # Shuffle indices for each epoch
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # print("Start and end indices for minibatch:", start, end)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]  # Calculate the log ratio
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approximate_kl = (-logratio).mean()
                    aproximate_kl = ((ratio-1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy gradient loss
                pg_loss1 = -mb_advantages * ratio  
                pg_loss2 = -mb_advantages * torch.clamp(
                                                        ratio, 
                                                        1 - args.clip_coef, 
                                                        1 + args.clip_coef
                                                        )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value function loss
                if args.clip_vloss: # Clip the value loss from Paper
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], 
                        -args.clip_coef, 
                        args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # Normal MSE loss

                entropy_loss = entropy.mean()  # Entropy loss for exploration
                # Minimize the policy loss function and the value loss function
                # Maximize the entropy loss
                loss = (pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss) 

                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backpropagate the loss
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # Clip the gradients
                optimizer.step()  # Update the parameters
            
            
            if args.target_kl is not None and old_approximate_kl > args.target_kl:
                break

        # Explain the variance of the predictions
        # Indicate if the variance is a good indicator of the returns
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_pred - y_true) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approximate_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", aproximate_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # Final video logging
    if args.track and args.capture_video:
        print("final videos to wandb...")
        log_videos_to_wandb(video_dir, global_step)
