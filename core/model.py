import torch
import torch.nn as nn
 
 
class DRQN(nn.Module):
    def __init__(self, n_observations=8, n_actions=5):
        super(DRQN, self).__init__()
        # Feature extraction
        self.fc1 = nn.Linear(n_observations, 64)
        # Recurrent layer (LSTM) để nhớ lịch sử metrics
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        # Output layer
        self.fc2 = nn.Linear(128, n_actions)
 
    def forward(self, x, hidden=None, return_all: bool = False):
        # x shape: (batch, seq_len, features)
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
 
        if return_all:
            # Training mode: trả về Q-values cho tất cả timestep → (batch, seq_len, n_actions)
            x = self.fc2(x)
        else:
            # Inference mode: chỉ lấy output của timestep cuối → (batch, n_actions)
            x = self.fc2(x[:, -1, :])
 
        return x, hidden