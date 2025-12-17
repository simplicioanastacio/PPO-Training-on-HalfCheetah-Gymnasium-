import gymnasium as gym

# (Mujoco version)
env = gym.make("HalfCheetah-v4", render_mode="human")

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  
env.close()