import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from torch_geometric.nn import GCNConv, SAGEConv
import args


class GraphVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphVAE, self).__init__()
        # Define the GCN layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv_mu = SAGEConv(hidden_channels, out_channels)
        self.conv_logvar = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index, edge_weight, node_labels):
        x = x.float()  # Ensure x is of type Float
        num_node_labels = node_labels.max().item() + 1
        x = F.one_hot(node_labels, num_classes=num_node_labels).float()
        edge_index = edge_index.long()  # Ensure edge_index is of type Long
        edge_weight = edge_weight.float()  # Ensure edge_weight is of type Float
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        mu = self.conv_mu(x, edge_index, edge_weight)
        logvar = self.conv_logvar(x, edge_index, edge_weight)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, data):
        mu, logvar = self.encode(data.x, data.edge_index, data.edge_attr)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, data.edge_index), mu, logvar

    def decode(self, z, edge_index):
        return torch.sigmoid(torch.mm(z, z.t()))

    def recon_loss(self, z, pos_edge_index, neg_edge_index):
        pos_loss = -torch.log(
            self.decode(z, pos_edge_index) + 1e-15
        ).mean()
        neg_loss = -torch.log(
            1 - self.decode(z, neg_edge_index) + 1e-15
        ).mean()
        return pos_loss + neg_loss

    def kl_loss(self, mu, logvar):
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

