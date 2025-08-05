import argparse
import os
import random
import time
from distutils.util import strtobool
import numpy as np
import torch
import glob
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
import wandb
import gymnasium as gym  # Using gymnasium instead of gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from gymnasium.wrappers import RecordVideo, TimeLimit


def parse_args():
    parser = argparse.ArgumentParser(description='PPO Training Script')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rsplit('.py')[0],
                        help='Name of the experiment') # Ecperiment name, value of the file name without extension
    parser.add_argument('--gym-id', type=str, default='MiniGrid-Empty-5x5-v0',
                        help='ID of the Gym environment to use')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed of the experiment for reproducibility')
    parser.add_argument('--total-timesteps', type=int, default=500000,
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
    parser.add_argument('--num-steps', type=int, default=1024,
                        help='number of steps to run in each environment per update') # Contols how much data is collected before updating the agent
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=False,
                        help='if toggled, the learning rate will be annealed to zero') # Anneal the learning rate to zero over the course of training
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, use Generalized Advantage Estimation (GAE)') # Use GAE for advantage estimation
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards') # Discount factor for rewards
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='lambda for GAE') # Lambda for GAE, controls the bias-variance tradeoff
    parser.add_argument('--num-minibatches', type=int, default=4,
                        help='number of mini-batches to split the data into for each update') # Number of mini-batches to split the data into for each update
    parser.add_argument('--num-epochs', type=int, default=8,
                        help='number of epochs to update policy') # Number of epochs to update the agent for each update
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, normalize the advantages') # Normalize the advantages to have zero mean and unit variance  
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help='clipping coefficient for PPO') # Clipping coefficient for PPO, controls the policy
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, clip the value loss') # Clip the value loss to prevent large updates
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='coefficient for the entropy bonus') # Coefficient for the entropy bonus, encourages exploration
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='coefficient for the value function loss') # Coefficient for the value function loss, controls the importance of the value function
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='maximum norm for the gradients') # Maximum norm for the gradients, prevents exploding gradients
    parser.add_argument('--target-kl', type=float, default=None,
                        help='target KL divergence for early stopping') # Target KL divergence for early stopping, prevents
    parser.add_argument('--frame-stack', type=int, default=4, 
                        help='how many last RGB frames to stack')

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
        if "MiniGrid" in gym_id:
            # Create MiniGrid environment with specific wrappers
            env = gym.make(gym_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
            env = RGBImgObsWrapper(env)  # Get RGB observations
            env = ImgObsWrapper(env)      # Get only image observations
            
            # Get the current observation space
            old_obs_space = env.observation_space
            
            # Create new observation space with transposed dimensions
            new_obs_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(3, old_obs_space.shape[0], old_obs_space.shape[1]),
                dtype=np.uint8
            )
            
            # Add permute wrapper with the new observation space
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: obs.transpose(2, 0, 1),
                observation_space=new_obs_space
            )
        else:
            # Create regular gym environment
            env = gym.make(gym_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        
        # Add video recording for first environment
        if capture_video and idx == 0:
            env = RecordVideo(
                env,
                f'videos/{run_name}',
                episode_trigger=lambda ep: True,
                video_length=200
            )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        print(env.action_space, env.observation_space)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
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
        super().__init__()
        # Input shape will be (N, 3, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the shape of flattened features
        # Use the actual observation space shape from the environment
        with torch.no_grad():
            sample_obs = torch.zeros(1, *envs.single_observation_space.shape)
            conv_out_size = self.conv(sample_obs).shape[1]

        self.actor = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(conv_out_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )

    def get_value(self, x):
        features = self.conv(x / 255.0)  # Normalize pixel values
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.conv(x / 255.0)  # Normalize pixel values
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


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
    print(envs.single_observation_space, envs.single_action_space)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "Only supports discrete action spaces for now"

    agent = Agent(envs).to(device)  # Initialize the agent
    print(agent)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Optimizer for the agent, epsilon same as paper

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) # Collect observations + how many env
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}, device: {obs.device}")
    actions = torch.zeros(
                (args.num_steps, args.num_envs) + envs.single_action_space.shape,
                dtype=torch.long,           # <- important
            ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    print(f"actions shape: {actions.shape}, dtype: {actions.dtype}, device: {actions.device}")
    print(f"logprobs shape: {logprobs.shape}, dtype: {logprobs.dtype}, device: {logprobs.device}")
    print(f"rewards shape: {rewards.shape}, dtype: {rewards.dtype}, device: {rewards.device}")
    print(f"dones shape: {dones.shape}, dtype: {dones.dtype}, device: {dones.device}")
    print(f"values shape: {values.shape}, dtype: {values.dtype}, device: {values.device}")
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device) # Reset the environment and get the initial observation
    next_done  = torch.zeros(args.num_envs).to(device) # Initialize dones
    print(f"next_obs shape: {next_obs.shape}, next_done shape: {next_done.shape}")

    for update in range(args.num_updates):
        # Anneal the learning rate if specified
        if args.anneal_lr:
            frac = 1.0 - update / args.num_updates # Linear annealing lr to zero
            lr = args.learning_rate * frac # Annealed learning rate
            optimizer.param_groups[0]['lr'] = lr # Update the learning rate in the optimizer

        for step in range(0, args.num_steps): # Collect data for each step Rollout
            global_step += args.num_envs  # Increment global step by the number of environments
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs) # Get the value of the state
                values[step] = value.view(-1) # Store the value of the state
            
            actions[step] = action.long()
            logprobs[step] = logprob

            next_obs, reward, done, truncated, infos = envs.step(action.cpu().numpy())  # Step the environment
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # Store the reward
            next_obs = torch.tensor(next_obs).to(device)
            next_done = torch.as_tensor(
                np.logical_or(done, truncated),
                device=device,
                dtype=torch.float32,
            )  # Update next_obs and next_done

        
            if 'episode' in infos:
                # Calculate mean reward across all environments
                r = np.mean(infos['episode']['r'])
                l = np.mean(infos['episode']['l'])
                print(f"global_step: {global_step}, episode reward: {r}")
                
                writer.add_scalar('charts/episodic_return', r, global_step)  # Log the mean episodic return
                writer.add_scalar('charts/episodic_length', l, global_step)  # Log the mean episodic length
                #if args.track and args.capture_video:
                #    log_videos_to_wandb(video_dir, global_step)

    
        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)
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

                

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]  # Calculate the log ratio
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approximate_kl = (-logratio).mean()
                    approximate_kl = ((ratio-1) - logratio).mean()
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
            
            
            #if args.target_kl is not None and old_approximate_kl > args.target_kl:
            #    break

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
        writer.add_scalar("losses/approx_kl", approximate_kl.item(), global_step)
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