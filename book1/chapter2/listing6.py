# Solving the n-armed bandit
import matplotlib.pyplot as plt
import numpy as np
import random


def get_best_arm(record):
    arm_index = np.argmax(record[:, 1], axis=0)
    return arm_index


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
record = np.zeros((n, 2))
probs = np.random.rand(n)
eps = 0.2
rewards = [0]

for i in range(500):
    if random.random() > eps:
        choice = get_best_arm(record)
    else:
        choice = np.random.randint(10)
    r = get_reward(probs[choice])
    record = update_record(record, choice, r)
    mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
    rewards.append(mean_reward)

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
ax.scatter(np.arange(len(rewards)), rewards)
plt.show()
