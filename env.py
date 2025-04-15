import numpy as np


# Define the bandit environment
class BanditEnv:
    def __init__(self, conflict):
        self.conflict = conflict

    def get_trial(self):
        """Generate a single trial (one 9-dimensional input)."""
        state = np.random.choice([0, 1])
        obj1 = np.eye(4)[np.random.choice(4)]
        obj2 = np.eye(4)[np.random.choice(4)]
        obs = np.concatenate([[state], obj1, obj2])
        return obs, self.get_reward(state, obj1, obj2)

    def get_reward(self, state, obj1, obj2):
        """Compute rewards based on state and objects."""
        if self.conflict:
            rewards = np.array([1, 2, 3, 4]) if state == 0 else np.array([4, 3, 2, 1])
        else:
            rewards = np.array([1, 2, 3, 4])
        return np.dot(obj1, rewards), np.dot(obj2, rewards)  # (reward for left, reward for right)
    