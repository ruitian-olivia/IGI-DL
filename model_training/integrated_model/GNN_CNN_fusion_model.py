import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GINConv, GCNConv, GATConv, BatchNorm, PNAConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

from MLP_model import MLP
from ViT_model import ViT_extractor

class GIN4layer_ResNet18(nn.Module):
    def __init__(self, num_feature, num_gene, nhid=512, mlp_hidden_list=[256,256]):
        super(GIN4layer_ResNet18, self).__init__()

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

        return y

class GCN4layer_ResNet18(nn.Module):
    def __init__(self, num_feature, num_gene, nhid=512, mlp_hidden_list=[256,256]):
        super(GCN4layer_ResNet18, self).__init__()

        # ResNet18
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.resnet_BN = nn.BatchNorm1d(512)

        # GCN
        self.nhid = nhid
        self.gcn_conv1 = GCNConv(int(num_feature), self.nhid)
        self.gcn_bn1   = BatchNorm(self.nhid)
        self.gcn_conv2 = GCNConv(self.nhid, self.nhid)
        self.gcn_bn2   = BatchNorm(self.nhid)
        self.gcn_conv3 = GCNConv(self.nhid, self.nhid)
        self.gcn_bn3   = BatchNorm(self.nhid)
        self.gcn_conv4 = GCNConv(self.nhid, self.nhid)
        self.gcn_bn4   = BatchNorm(self.nhid)

        self.gcn_lin1 = nn.Linear(self.nhid*2, self.nhid)
        self.gcn_lin2 = nn.Linear(self.nhid, self.nhid//2)
        self.gcn_BN = nn.BatchNorm1d(self.nhid//2)

        # Fusion
        fusion_mlp = []
        in_features = self.nhid//2+512
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
        
        # GCN
        graph_x = F.relu(self.gcn_bn1(self.gcn_conv1(graph_x, edge_index)))
        graph_x1 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = F.relu(self.gcn_bn2(self.gcn_conv2(graph_x, edge_index)))
        graph_x2 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = F.relu(self.gcn_bn3(self.gcn_conv3(graph_x, edge_index)))
        graph_x3 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)
        
        graph_x = F.relu(self.gcn_bn4(self.gcn_conv4(graph_x, edge_index)))
        graph_x4 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = graph_x1 + graph_x2 + graph_x3 + graph_x4

        graph_x = F.relu(self.gcn_lin1(graph_x))
        graph_x = F.dropout(graph_x, p=0.5, training=self.training)
        graph_out = self.gcn_BN(self.gcn_lin2(graph_x))
        
        # Fusion
        concate_x = torch.cat((img_output, graph_out), 1)
        fusion_x = self.fusion_mlp(concate_x)
        y = torch.squeeze(self.head(fusion_x), dim=1)

        return y

class GAT4layer_ResNet18(nn.Module):
    def __init__(self, num_feature, num_gene, nhid=512, mlp_hidden_list=[256,256]):
        super(GAT4layer_ResNet18, self).__init__()

        # ResNet18
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.resnet_BN = nn.BatchNorm1d(512)

        # GAT
        self.gat_conv1 = GATConv(num_feature, nhid)
        self.gat_bn1   = BatchNorm(nhid)
        self.gat_conv2 = GATConv(nhid, nhid)
        self.gat_bn2   = BatchNorm(nhid)
        self.gat_conv3 = GATConv(nhid, nhid)
        self.gat_bn3   = BatchNorm(nhid)
        self.gat_conv4 = GATConv(nhid, nhid)
        self.gat_bn4   = BatchNorm(nhid)

        self.gat_lin1 = torch.nn.Linear(2*nhid, nhid)
        self.gat_lin2 = torch.nn.Linear(nhid, nhid//2)
        self.gat_BN = nn.BatchNorm1d(nhid//2)

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
        
        # GAT
        graph_x = F.relu(self.gat_bn1(self.gat_conv1(graph_x, edge_index)))
        graph_x1 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = F.relu(self.gat_bn2(self.gat_conv2(graph_x, edge_index)))
        graph_x2 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = F.relu(self.gat_bn3(self.gat_conv3(graph_x, edge_index)))
        graph_x3 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)
        
        graph_x = F.relu(self.gat_bn4(self.gat_conv4(graph_x, edge_index)))
        graph_x4 = torch.cat([gmp(graph_x, batch), gap(graph_x, batch)], dim=1)

        graph_x = graph_x1 + graph_x2 + graph_x3 + graph_x4

        graph_x = F.relu(self.gat_lin1(graph_x))
        graph_x = F.dropout(graph_x, p=0.5, training=self.training)
        graph_out = self.gat_BN(self.gat_lin2(graph_x))
        
        # Fusion
        concate_x = torch.cat((img_output, graph_out), 1)
        fusion_x = self.fusion_mlp(concate_x)
        y = torch.squeeze(self.head(fusion_x), dim=1)

        return y
    
class GIN4layer_ViT(nn.Module):
    def __init__(self, num_feature, num_gene, nhid=256, attn_heads=8, dim_head=64, hidden_features=256, out_features=256, mlp_hidden_list=[256,256]):
        super(GIN4layer_ViT, self).__init__()

        # ViT
        self.ViT = ViT_extractor(attn_heads=attn_heads, dim_head=dim_head, hidden_features=hidden_features, out_features=out_features)
        self.ViT_BN = nn.BatchNorm1d(out_features)

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
        in_features = out_features + nhid//2
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
        
        # ViT
        img_output = self.ViT_BN(self.ViT(data.patch_img)) # Input image: B x 3 x 200 x 200
        
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

        return y