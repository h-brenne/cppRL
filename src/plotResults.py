from email.utils import decode_rfc2231
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df11 = pd.read_csv("run-end-rewards-coverage_percentage.csv")
df12 = pd.read_csv("run-end-rewards-episode_reward.csv")
df13 = pd.read_csv("run-end-rewards-overlap_percentage.csv")
df21 = pd.read_csv("run-step-rewards-coverage_percentage.csv")
df22 = pd.read_csv("run-step-rewards-episode_reward.csv")
df23 = pd.read_csv("run-step-rewards-overlap_percentage.csv")
df3 = pd.read_csv("run-random-episode_reward.csv")

fig = plt.figure()
"""
plt.plot(df22["Step"][0:500], df22["Value"][0:500], alpha=0.2, label="Training")
plt.plot(df22["Step"][0:500], df22.ewm(alpha=(1-0.85)).mean()["Value"][0:500], label="Training smoothed")
plt.plot(df3["Step"], df3.ewm(alpha=(1-0.85)).mean()["Value"], label="Random actions smoothed")
plt.title("Episode reward")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
"""

plt.subplot(1,2,1)
plt.plot(df11["Step"], df11.ewm(alpha=(1-0.85)).mean()["Value"], label="End reward")
plt.plot(df21["Step"][0:500], df21.ewm(alpha=(1-0.85)).mean()["Value"][0:500], label="Step reward")

plt.title("Coverage Percentage")
plt.xlabel("Step")
plt.ylabel("Percentage")
plt.legend()

plt.subplot(1,2,2)

plt.plot(df13["Step"], df13.ewm(alpha=(1-0.85)).mean()["Value"], label="End reward")
plt.plot(df23["Step"][0:500], df23.ewm(alpha=(1-0.85)).mean()["Value"][0:500], label="Step reward")

plt.title("Overlap Percentage")
plt.xlabel("Step")
plt.ylabel("Percentage")
plt.legend()


plt.show()