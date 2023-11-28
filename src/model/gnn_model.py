import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class CNN(torch.nn.Module):
    def __init__(self, n_input, n_embed = 768):
        super(CNN, self).__init__()
        self.conv1 = GCNConv(n_input, 2048)
        self.conv2 = GCNConv(2048, 2048)
        self.conv3 = GCNConv(2048, 1024)
        self.linear = Linear(1024, n_embed)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).tanh()
        h = self.conv2(h, edge_index).tanh()
        h = self.conv3(h, edge_index).tanh()
        out = self.linear(h)
        return out
