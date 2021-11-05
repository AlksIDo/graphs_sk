import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool


class NonGraphModel(nn.Module):
    def __init__(self):
        super(NonGraphModel, self).__init__()
        torch.manual_seed(42)
        self.node1 = Linear(27, 30)
        self.node2 = Linear(30, 30)
        self.node3 = Linear(30, 1)

        self.lin1 = Linear(538, 250)
        self.lin2 = Linear(250, 125)
        self.lin3 = Linear(125, 50)
        self.lin4 = Linear(50, 1)

    def forward(self, x):
        # 1. Obtain node embeddings

        x = self.node1(x)
        x = x.relu()
        x = self.node2(x)
        x = x.relu()
        x = self.node3(x)

        x = torch.flatten(x, start_dim=1)

        emb = self.lin1(x)
        emb = emb.relu()
        emb = self.lin2(emb)
        emb = emb.relu()
        emb = self.lin3(emb)
        emb = emb.relu()
        emb = self.lin4(emb)

        return emb


class DGLRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(DGLRegressor, self).__init__()
        torch.manual_seed(42)
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv3 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv4 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h.double()))
        h = F.relu(self.conv2(g, h.double()))
        h = F.relu(self.conv3(g, h.double()))
        h = F.relu(self.conv4(g, h.double()))

        with g.local_scope():
            g.ndata["h"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "h")

            hg = self.linear1(hg)
            hg = F.relu(hg)
            hg = self.linear2(hg)
            hg = F.relu(hg)

            return self.out(hg)


class PGRegressor(torch.nn.Module):
    def __init__(self, hidden_channels, node_feature_channels):
        super(PGRegressor, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GraphConv(node_feature_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)

    def forward(self, data):
        # 1. Obtain node embeddings
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()

        return self.lin3(x)
