import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): 
        return self.net(x)