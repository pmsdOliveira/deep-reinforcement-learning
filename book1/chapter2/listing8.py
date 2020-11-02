# Softmax action-selection for the n-armed bandit
import matplotlib.pyplot as plt
import numpy as np
import random


def softmax(av, tau=1.12):
    softm = np.exp(av / tau) / np.sum(np.exp(av / tau))
    return softm


def get_reward(prob, n=10):
    reward = 0
    for _ in range(n):
        if random.random() < prob:
            reward += 1
    return reward


def update_record(record, action, r):
    new_r = (record[action, 0] * record[action, 1] + r) / \
        (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_r
    return record


n = 10
probs = np.random.rand(n)
record = np.zeros((n, 2))
rewards = [0]

for i in range(500):
    p = softmax(record[:, 1])
    choice = np.random.choice(np.arange(n), p=p)
    r = get_reward(probs[choice])
    record = update_record(record, choice, r)
    mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
    rewards.append(mean_reward)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9, 5)
ax.scatter(np.arange(len(rewards)), rewards)
plt.show()
