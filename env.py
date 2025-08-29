import numpy as np


# Define the bandit environment
class BanditEnv: 
    def __init__(self, conflict=True):
        self.conflict = conflict

    def get_trial(self):
        # Cue
        cue = np.random.choice([0, 1, 2])
        cue_onehot = np.eye(3)[idx_cue]
        """Generate a single trial (one 19 (3 + 8 * 2) -dimensional input)."""
        # stimulus set
        if idx_cue == 0 or idx_cue == 1:
            stimulus_set = [1, 2, 3, 4]
        elif idx_cue == 2:
            stimulus_set = [4, 5, 6, 7]
        
        # obj1 != obj2
        idx1 = np.random.choice(stimulus_set)
        idx2 = np.random.choice([i for i in stimulus_set if i != idx1])
        obj1 = np.eye(8)[idx1]
        obj2 = np.eye(8)[idx2]
        obs = np.concatenate([cue_onehot, obj1, obj2])
        return obs, self.get_reward(cue, obj1, obj2)

    def get_reward(self, cue, obj1, obj2):
        """Compute rewards based on state and objects."""
        if self.conflict:
            if cue == 1:
                rewards = np.array([4, 3, 2, 1, 4, 3, 2, 1]) 
            else:
                rewards = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        else:
            rewards = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        return np.dot(rewards, obj1), np.dot(rewards, obj2)  # (reward for left, reward for right)
    