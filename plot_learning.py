import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load training logs
log_file = "./ppo_halfcheetah_logs/monitor.csv"
df = pd.read_csv(log_file, skiprows=1) #skip first line(metadata)

#Extract episodes and rewards
episodes = np.arange(len(df))
rewards = df["r"].rolling(window=10).mean()

#Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, label="Average Reward (10 episodes)")
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
plt.title("PPO Learning Curve for Halfcheetah-v4")
plt.legend()
plt.grid(True)
plt.savefig("ppo_halfcheetah_learning_curve.png")
plt.show()