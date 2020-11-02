# Finding the best actions given the expected rewards
def get_best_action(actions):
    best_action = 0
    max_action_value = 0
    for i in range(len(actions)):
        curr_action_value = actions[i]
        if curr_action_value > max_action_value:
            best_action = i
            max_action_value = curr_action_value
    return best_action
