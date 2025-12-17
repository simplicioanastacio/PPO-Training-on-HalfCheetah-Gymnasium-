import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "./ppo_halfcheetah_logs"
os.makedirs(log_dir, exist_ok = True)

#Create environment
env = gym.make("HalfCheetah-v4")
env = Monitor(env, filename = os.path.join(log_dir, "monitor.csv")) 

model = PPO(
    policy = "MlpPolicy",
    env = env,
    learning_rate = 2.8e-4, #slightly lower LR for more stable training
    n_steps = 2048,
    batch_size = 128,
    gamma = 0.99, #discount factor for future rewards
    verbose = 1,
    tensorboard_log = log_dir
)

model.learn(total_timesteps = 100000)

model.save("ppo_halfcheetah")

env.close()