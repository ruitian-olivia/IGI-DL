import torch
import torch.nn as nn
from torchvision import models

from MLP_model import MLP

class ResNet_MLP_gene(nn.Module):
    def __init__(self, gene_dim, mlp_hidden_list=[256,256], model_type='resnet18'):
        super(ResNet_MLP_gene, self).__init__()
        if model_type=='resnet18':
            resnet = models.resnet18(pretrained=False)
        elif model_type=='resnet34':
            resnet = models.resnet34(pretrained=False)
        else:
            print("error in choosing the type of ResNet!")
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        mlp_block = []
        in_features = 512
        for per_hidden_features in mlp_hidden_list:
            mlp_block.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    dropout_prob=0.5,
                )
            )
            in_features = per_hidden_features

        self.mlp = nn.Sequential(*mlp_block)
        self.head = nn.Linear(in_features, gene_dim)

    def forward(self, x):

        resnet_output = self.resnet(x)  
        x1 = resnet_output.view(resnet_output.size(0), -1)  
        x1 = self.mlp(x1)
        y = torch.squeeze(self.head(x1), dim=1)

        return y
