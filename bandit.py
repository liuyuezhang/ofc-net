import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Q-learning parameters
gamma = 0.99  # Discount factor (not needed in bandit, but kept for consistency)
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
lr = 0.001
num_trials = 10000  # Number of independent trials
conflict = False
log_interval = 1000  # Log the average reward every 1000 trials

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment and model
from env import BanditEnv
env = BanditEnv(conflict=conflict)

from model import QNetwork
q_network = QNetwork().to(device)  # Move model to GPU if available
optimizer = optim.Adam(q_network.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Variables for logging average reward
total_rewards = []  # To track total reward for each trial

# Training loop
for trial in range(num_trials):
    state, (reward_left, reward_right) = env.get_trial()
    state_tensor = torch.FloatTensor(state).to(device)

    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1])
    else:
        with torch.no_grad():
            action = torch.argmax(q_network(state_tensor)).item()

    # Get actual reward based on action
    reward = reward_left if action == 0 else reward_right

    # Compute Q-learning update (no next state in bandit setting)
    q_values = q_network(state_tensor)
    target_q = torch.tensor(reward, dtype=torch.float32).to(device)

    loss = loss_fn(q_values[action], target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track rewards for average reward calculation
    total_rewards.append(reward)

    # Logging every log_interval trials
    if (trial + 1) % log_interval == 0:
        avg_reward = np.mean(total_rewards[-log_interval:])
        print(f"Trial {trial + 1}: Avg Reward = {avg_reward:.2f}, Epsilon = {epsilon:.3f}")

# Save the trained model
torch.save(q_network.state_dict(), "bandit_conflict={}.pth".format('conflict' if conflict else 'no-conflict'))

print("Training completed and model saved!")
