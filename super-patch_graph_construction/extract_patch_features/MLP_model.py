import torch
import torch.nn as nn
from typing import List, Optional

class Unit(nn.Module):
    """
    One MLP layer. 
    It orders the operations as: fc -> BN -> ReLU -> dropout
    """
    def __init__(self, in_features: int, out_features: int, dropout_prob: float):
        super(Unit, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.BN = nn.BatchNorm1d(out_features)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc(x)
        x = self.BN(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP). If the hidden or output feature dimension is
    not provided, we assign it the input feature dimension.
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            num_layers: Optional[int] = 1,
            dropout_prob: Optional[float] = 0.5,
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        layers = []
        for _ in range(num_layers):
            per_unit = Unit(
                in_features=in_features,
                out_features=hidden_features,
                dropout_prob=dropout_prob
            )
            in_features = hidden_features
            layers.append(per_unit)
        if out_features != hidden_features:
            self.fc_out = nn.Linear(hidden_features, out_features)
        else:
            self.fc_out = None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.fc_out is not None:
            return self.fc_out(x)
        else:
            return x