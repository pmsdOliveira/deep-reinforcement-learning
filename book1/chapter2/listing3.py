# Defining the reward function
import numpy as np
import random
import matplotlib.pyplot as plt


def get_reward(prob, n=10):
    reward = 0
    for _ in range(n):
        if random.random() < prob:
            reward += 1
    return reward


reward_test = [get_reward(0.7) for _ in range(2000)]
print(np.mean(reward_test))

plt.figure(figsize=(9, 5))
plt.xlabel("Reward", fontsize=22)
plt.ylabel("# Observations", fontsize=22)
plt.hist(reward_test, bins=9)
plt.show()
