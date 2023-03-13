import matplotlib.pyplot as plt
import numpy as np

GOAL = 100
STATE = np.arange(GOAL+1)
HEAD_PROB = 0.2

def value_iteration():
    state_value = np.zeros(GOAL+1)
    state_value[GOAL] = 1
    while True:
        old_state_value = state_value.copy()

        for state in STATE[1:GOAL]:
            actions = np.arange(min(state,GOAL-state)+1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action]+(1-HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        max_diff = abs(state_value-old_state_value).sum()
        print(max_diff)
        if max_diff < 1e-10:
            break

        policy = np.zeros(GOAL + 1)
        for state in STATE[1:GOAL]:
            actions = np.arange(min(state,GOAL-state)+1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action]+(1-HEAD_PROB)*state_value[state - action])
            # policy[state] = actions[np.argmax(action_returns[1:])+1]
            policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]
            print(np.round(action_returns[1:], 5))
    print(policy)

if __name__ == "__main__":
    value_iteration()