# Updating the reward record
import numpy as np


def update_record(record, action, r):
    new_r = (record[action, 0] * record[action, 1] + r) / \
        (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_r
    return record
