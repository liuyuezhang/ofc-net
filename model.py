import torch
import torch.nn as nn


# Define the MLP for action-value estimation
class QNetwork(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, output_dim=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     return self.fc3(x)  # Q-values for left and right actions
    
    def encode(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
    def decode(self, x):
        return self.fc3(x)

    def forward(self, x):
        r = self.encode(x)
        y = self.decode(r)
        return r, y  # y: Q-values for left and right actions
