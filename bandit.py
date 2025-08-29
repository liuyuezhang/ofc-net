import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


# Q-learning parameters
n = 64
coef_l1 = 0.0
gamma = 0.99  # Discount factor (not needed in bandit, but kept for consistency)
epsilon = 1.0
epsilon_min = 0.0
epsilon_decay = 0.995
lr = 0.001
num_trials = 10000  # Number of independent trials
conflict = True
log_interval = 1000  # Log the average reward every 1000 trials

# name
res_dir = './res'
name = 'bandit_conflict={}_n={}_l1={}'.format(conflict, n, coef_l1)
name_dir = os.path.join(res_dir, name)
os.makedirs(name_dir, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment and model
from env import BanditEnv
env = BanditEnv(conflict=conflict)

from model import QNetwork
q_network = QNetwork(hidden_dim=n).to(device)  # Move model to GPU if available
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
            r, y = q_network(state_tensor)
            action = torch.argmax(y).item()

    # Get actual reward based on action
    reward = reward_left if action == 0 else reward_right

    # Compute Q-learning update (no next state in bandit setting)
    r, y = q_network(state_tensor)
    target_q = torch.tensor(reward, dtype=torch.float32).to(device)

    # loss
    loss_main = loss_fn(y[action], target_q)
    loss_l1 = torch.abs(r).mean()
    lbd_l1 = coef_l1 * loss_main.item() / loss_l1.item()
    loss = loss_main + lbd_l1 * loss_l1

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
torch.save(q_network.state_dict(), name_dir + "/model.pth")

print("Training completed and model saved!")
