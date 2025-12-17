import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# Load trained model
model = PPO.load("ppo_halfcheetah")

# Create environment with video recording
env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="./halfcheetah_videos",
    name_prefix="ppo_halfcheetah",
    episode_trigger=lambda x: x == 0  # record first episode only
)

# Run trained agent
obs, info = env.reset()

for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Video saved to ./halfcheetah_videos/")
