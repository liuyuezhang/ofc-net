import numpy as np


# Define the bandit environment
class BanditEnv: 
    def __init__(self, conflict=True, p_force=0.25, num_cues=2):
        self.conflict = conflict
        self.p_force = p_force
        self.num_cues = num_cues

    def get_trial(self):
        # Cue
        cue = np.random.choice(self.num_cues)
        cue_onehot = np.eye(3)[cue]
        """Generate a single trial (one 19 (8 * 2 + 3) -dimensional input)."""
        # stimulus set
        if cue == 0 or cue == 1:
            stimulus_set = [0, 1, 2, 3]
        elif cue == 2:
            stimulus_set = [4, 5, 6, 7]
        
        if np.random.rand() < self.p_force:
            # single force choice (for reward 1)
            obj = np.eye(8)[np.random.choice(stimulus_set)]
            empty_obj = np.zeros(8)
            if np.random.rand() < 0.5:
                obj1 = obj
                obj2 = empty_obj
            else:
                obj1 = empty_obj
                obj2 = obj
        else:
            # obj1 != obj2
            idx1 = np.random.choice(stimulus_set)
            idx2 = np.random.choice([i for i in stimulus_set if i != idx1])
            obj1 = np.eye(8)[idx1]
            obj2 = np.eye(8)[idx2]
        obs = np.concatenate([obj1, obj2, cue_onehot])
        return obs, self.get_reward(obj1, obj2, cue)

    def get_reward(self, obj1, obj2, cue):
        """Compute rewards based on state and objects."""
        if self.conflict:
            if cue == 1:
                rewards = np.array([4, 3, 2, 1, 4, 3, 2, 1]) 
            else:
                rewards = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        else:
            rewards = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        return np.dot(rewards, obj1), np.dot(rewards, obj2)  # (reward for left, reward for right)
    