import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, hidden_dim, alloc_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim + alloc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # non-negative ROAS
        )

    def forward(self, h, alloc):
        """
        h: (batch_size, hidden_dim)
        alloc: (batch_size, alloc_dim)
        """
        x = torch.cat([h, alloc], dim=-1)
        return self.model(x).squeeze(-1)  # (batch_size,)


class Allocator(nn.Module):
    def __init__(self, hidden_dim, alloc_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim)
        )

    def forward(self, h):
        raw_alloc = self.model(h)  # (batch_size, alloc_dim)
        alloc = F.softmax(raw_alloc, dim=-1)  # Allocation as percentage vector
        return alloc
