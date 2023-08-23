import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GINConv, TopKPooling, GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from MLP_model import MLP

class GIN4layer_ResNet18_visual(nn.Module):
    def __init__(self, num_feature, num_gene, nhid=512, mlp_hidden_list=[256,256]):
        super(GIN4layer_ResNet18_visual, self).__init__()

        # ResNet18
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.resnet_BN = nn.BatchNorm1d(512)

        # GIN
        self.gin_conv1 = GINConv(Seq(Lin(num_feature, nhid), ReLU(), Lin(nhid, nhid)))
        self.gin_bn1 = BatchNorm(nhid)
        self.gin_conv2 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.gin_bn2 = BatchNorm(nhid)
        self.gin_conv3 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.gin_bn3 = BatchNorm(nhid)
        self.gin_conv4 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        self.gin_bn4 = BatchNorm(nhid)

        self.gin_lin1 = nn.Linear(2*nhid, nhid)
        self.gin_lin2 = nn.Linear(nhid, nhid//2)
        self.gin_BN = nn.BatchNorm1d(nhid//2)

        # Fusion
        fusion_mlp = []
        in_features = nhid//2+512
        for per_hidden_features in mlp_hidden_list:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    dropout_prob=0.5,
                )
            )
            in_features = per_hidden_features
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        self.head = nn.Linear(in_features, num_gene)


    def forward(self, data):
        graph_x, edge_index, edge_attr, patch_img, batch = data.x, data.edge_index, data.edge_attr, data.patch_img, data.batch
        
        # ResNet18
        img_x = self.resnet(data.patch_img) # Input image: B x 3 x 200 x 200
        img_output = self.resnet_BN(img_x.view(img_x.size(0), -1)) # B x 64
        
        # GIN
        graph_x = F.relu(self.gin_bn1(self.gin_conv1(graph_x, edge_index)))
        graph_x1 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = F.relu(self.gin_bn2(self.gin_conv2(graph_x, edge_index)))
        graph_x2 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = F.relu(self.gin_bn3(self.gin_conv3(graph_x, edge_index)))
        graph_x3 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)
        
        graph_x = F.relu(self.gin_bn4(self.gin_conv4(graph_x, edge_index)))
        graph_x4 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = graph_x1 + graph_x2 + graph_x3 + graph_x4

        graph_x = F.relu(self.gin_lin1(graph_x))
        graph_x = F.dropout(graph_x, p=0.5, training=self.training)
        graph_out = self.gin_BN(self.gin_lin2(graph_x))
        
        # Fusion
        concate_x = torch.cat((img_output, graph_out), 1)
        fusion_x = self.fusion_mlp(concate_x)
        y = torch.squeeze(self.head(fusion_x), dim=1)

        return torch.squeeze(fusion_x,dim=1), y
