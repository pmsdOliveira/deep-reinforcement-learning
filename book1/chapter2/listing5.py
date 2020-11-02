# Computing the best action
import numpy as np


def get_best_arm(record):
    arm_index = np.argmax(record[:, 1], axis=0)
    return arm_index
