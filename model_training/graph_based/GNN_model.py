import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, BatchNorm, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class GIN_4layer(nn.Module):
    def __init__(self, num_feature, num_gene, nhid=256):
        super(GIN_4layer, self).__init__()
        self.conv1 = GINConv(Seq(Lin(num_feature, nhid), ReLU(), Lin(nhid, nhid)))
        self.bn1 = BatchNorm(nhid) # batch normalization
        self.conv2 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.bn2 = BatchNorm(nhid) # batch normalization
        self.conv3 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.bn3 = BatchNorm(nhid) # batch normalization
        self.conv4 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.bn4 = BatchNorm(nhid) # batch normalization

        self.lin1 = torch.nn.Linear(2*nhid, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_gene)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        y = torch.squeeze(self.lin3(x), dim=1)

        return y

class GCN_4layer(torch.nn.Module):
    def __init__(self, num_feature, num_gene, nhid=256):
        super(GCN_4layer, self).__init__()
        self.nhid = nhid
        self.conv1 = GCNConv(int(num_feature), self.nhid)
        self.bn1   = BatchNorm(self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.bn2   = BatchNorm(self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.bn3   = BatchNorm(self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)
        self.bn4   = BatchNorm(self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, num_gene)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        y = torch.squeeze(self.lin3(x), dim=1)

        return y

class GAT_4layer(torch.nn.Module):
    def __init__(self, num_feature, num_gene, nhid=256):
        super(GAT_4layer, self).__init__()
        self.conv1 = GATConv(num_feature, nhid)
        self.bn1   = BatchNorm(nhid)
        self.conv2 = GATConv(nhid, nhid)
        self.bn2   = BatchNorm(nhid)
        self.conv3 = GATConv(nhid, nhid)
        self.bn3   = BatchNorm(nhid)
        self.conv4 = GATConv(nhid, nhid)
        self.bn4   = BatchNorm(nhid)

        self.lin1 = torch.nn.Linear(2*nhid, nhid)
        self.lin2 = torch.nn.Linear(nhid, nhid//2)
        self.lin3 = torch.nn.Linear(nhid//2, num_gene)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        y = torch.squeeze(self.lin3(x), dim=1)

        return y