import torch
import torch.nn as nn

class LinkPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, weighted = False):
        super(LinkPredictorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim*2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.out = nn.ReLU() if weighted else nn.Sigmoid()

    def forward(self, h_u, h_v):
        h_concat = torch.cat([h_u, h_v], dim=-1)
        h_hidden = self.relu(self.fc1(h_concat))
        link_score = self.out(self.fc2(h_hidden))
        return link_score
