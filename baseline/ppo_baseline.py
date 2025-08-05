
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def make_env():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    return env

# DummyVecEnv expects a function that creates the env
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)  # Convert HWC to CHW for CnnPolicy

# Train PPO
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_minigrid")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
