import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_res_blocks=2, dropout=0.5):
        """
        A general classifier head using a linear layer, a series of residual blocks,
        and a final output layer.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            out_dim (int): Number of output classes.
            num_res_blocks (int): Number of residual blocks.
            dropout (float): Dropout probability.
        """
        super(Classifier, self).__init__()
        layers = []
        # Initial linear mapping with BN and ReLU
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Add residual blocks
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))

        # Final linear layer for classification
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
