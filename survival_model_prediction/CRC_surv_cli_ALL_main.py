# -*- coding: utf-8 -*-
import os
import sys
import argparse
from CRC_surv_cli_ALL_train import Train
import torch

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")

    parser.add_argument("--model_name", default="CRC_GAT_surv_clinical", help="Model name", type=str)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.0001, help="Weight decay rate", type=float)
    parser.add_argument("--num_epochs", default=100, help="Number of epochs", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--node_feature_num", default=157, help="Number of node features in constructed Super-node Graph", type=int)
    parser.add_argument("--clinical_num", default=4, help="Number of encoded clinical features", type=int)
    parser.add_argument("--initial_dim", default=50, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--FF_number", default=5, help="Selecting number of set for cross validation", type=int)
    parser.add_argument("--gpu", default=0, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)
    parser.add_argument("--random_seed", default=7, help="The random seed", type=int)

    Argument = parser.parse_args(args=[])

    return parser.parse_args()

def main():
    Argument = Parser_main()

    Train(Argument)

if __name__ == "__main__":
    main()
