import os
import torch
import multiprocessing
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.utils as g_util
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from sklearn.metrics import pairwise_distances
from torch_geometric.transforms import Polar

superpatch_dir = './preprocessed_WSI/supernode_graph/READ'
features_dir = './extract_patch_features/IGI_DL_READ'

predict_gene_path = '../model_weights/CRC/CRC_gene_list.txt'
with open(predict_gene_path, "r", encoding="utf-8") as f:
    predict_gene_list = f.read().splitlines()
    
# Load the gene names with high Pearson correlation in IGI_DL prediction in CRC
high_gene_path = './extract_patch_features/node_features_name/CRC_gene_features.txt'
with open(predict_gene_path, "r", encoding="utf-8") as f:
    high_gene_list = f.read().splitlines()
num_high_gene = len(high_gene_list)

print("len(high_gene_list):", num_high_gene)

sample_list = os.listdir(features_dir)

# for sample_name in os.listdir(features_dir):
def supernode_graph_construct(sample_name):
    gene_feature_path = os.path.join(features_dir, sample_name, 'gene_latent.csv')
    save_path = os.path.join(superpatch_dir, sample_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    gene_feature_df = pd.read_csv(gene_feature_path)
    gene_feature_df.columns = predict_gene_list + ['barcodes', 'X', 'Y']
    
    high_gene_feature_df = gene_feature_df.loc[:, ['X', 'Y'] + high_gene_list]
    
    graph_dataframe = high_gene_feature_df.sort_values(by = ['Y', 'X'])
    graph_dataframe = graph_dataframe.reset_index(drop = True)
    
    coordinate_df = graph_dataframe.iloc[:,0:2]
    
    index = list(graph_dataframe.index)
    graph_dataframe.insert(0,'index_orig', index)
    
    node_dict = {}
    for i in range(len(coordinate_df)):
        node_dict.setdefault(i,[])
        
    X = max(set(np.squeeze(graph_dataframe.loc[:, ['X']].values,axis = 1)))
    Y = max(set(np.squeeze(graph_dataframe.loc[:, ['Y']].values, axis = 1)))

    gridNum = 4
    X_size = int(X / gridNum)
    Y_size = int(Y / gridNum)
    
    threshold = 0.75
    
    with tqdm(total=(gridNum+2)*(gridNum+2)) as pbar:
        for p in range(gridNum+2):
            for q in range(gridNum+2):
                if p == 0 :
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                    elif q == (gridNum+1):
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                    else:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= 0
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                elif p == (gridNum+1) :
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum+1):
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                else :
                    if q == 0:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= 0
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif q == (gridNum+1):
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else:
                        is_X = graph_dataframe['X'] <= X_size * (p+1)
                        is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                        is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                        is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                        X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                if len(X_10) == 0:
                    pbar.update()
                    continue

                coordinate_dataframe = X_10.loc[:, ['X','Y']]
                X_10 = X_10.reset_index(drop = True)
                coordinate_list = coordinate_dataframe.values.tolist()
                index_list = coordinate_dataframe.index.tolist()

                feature_dataframe = X_10[X_10.columns.difference(['index_orig','X','Y'])]
                feature_list = feature_dataframe.values.tolist()
                coordinate_matrix = euclidean_distances(coordinate_list, coordinate_list)
                coordinate_matrix = np.where(coordinate_matrix > 2.9, 0 , 1) # 2.9 [2*np.sqrt(2)]
                cosine_matrix = cosine_similarity(feature_list, feature_list)

                Adj_list = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= threshold).astype(int)

                for c, item in enumerate(Adj_list):
                    for node_index in np.array(index_list)[item.astype('bool')]:
                        if node_index == index_list[c]:
                            pass
                        else:
                            node_dict[index_list[c]].append(node_index)


                pbar.update()
                
    dict_len_list = []

    for i in range(0, len(node_dict)):
        dict_len_list.append(len(node_dict[i]))

    arglist_strict = np.argsort(np.array(dict_len_list))
    arglist_strict = arglist_strict[::-1]

    for arg_value in arglist_strict:
        if arg_value in node_dict.keys():
            for adj_item in node_dict[arg_value]:
                if adj_item in node_dict.keys():
                    node_dict.pop(adj_item)
                    arglist_strict=np.delete(arglist_strict, np.argwhere(arglist_strict == adj_item))

    for key_value in node_dict.keys():
        node_dict[key_value] = list(set(node_dict[key_value]))

    supernode_coordinate_x_strict = []
    supernode_coordinate_y_strict = []
    supernode_feature_strict = []

    supernode_relate_value = [supernode_coordinate_x_strict,
                            supernode_coordinate_y_strict,
                            supernode_feature_strict]

    whole_feature = graph_dataframe[graph_dataframe.columns.difference(['index_orig','X','Y'])]

    with tqdm(total = len(node_dict.keys())) as pbar_node:
        for key_value in node_dict.keys():
            supernode_relate_value[0].append(graph_dataframe['X'][key_value])
            supernode_relate_value[1].append(graph_dataframe['Y'][key_value])
            if len(node_dict[key_value]) == 0:
                select_feature = whole_feature.iloc[key_value]
            else:
                select_feature = whole_feature.iloc[node_dict[key_value] + [key_value]]
                select_feature = select_feature.mean()
            if len(supernode_relate_value[2]) == 0:
                temp_select = np.array(select_feature)
                supernode_relate_value[2] = np.reshape(temp_select, (1,len(high_gene_list)))
            else:
                temp_select = np.array(select_feature)
                supernode_relate_value[2] = np.concatenate((supernode_relate_value[2], np.reshape(temp_select, (1,len(high_gene_list)))), axis=0)
            pbar_node.update()

    spatial_threshold = 5.5
    coordinate_integrate = pd.DataFrame({'X':supernode_relate_value[0],'Y':supernode_relate_value[1]})
    coordinate_matrix1 = euclidean_distances(coordinate_integrate, coordinate_integrate)
    coordinate_matrix1 = np.where(coordinate_matrix1 > spatial_threshold , 0 , 1)

    fromlist = []
    tolist = []

    with tqdm(total = len(coordinate_matrix1)) as pbar_pytorch_geom:
        for i in range(len(coordinate_matrix1)):
            temp = coordinate_matrix1[i,:]
            selectindex = np.where(temp > 0)[0].tolist()
            for index in selectindex:
                fromlist.append(int(i))
                tolist.append(int(index))
            pbar_pytorch_geom.update()

    edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
    x = torch.tensor(supernode_relate_value[2], dtype=torch.float)
    coor_pos = coordinate_integrate[['X', 'Y']].to_numpy()
    coor_pos = torch.tensor(coor_pos)
    graph = Data(x=x, edge_index=edge_index, coor_pos=coor_pos)

    # node_dict = pd.DataFrame.from_dict(node_dict, orient='index')
    # node_dict.to_csv(os.path.join(superpatch_dir, "test" + '_' + str(threshold) + '.csv'))
    torch.save(graph, os.path.join(save_path, str(threshold) + '_graph_torch.pt'))
    
    distance_thresh = 4.3 # 3*np.sqrt(2)
    coordinate_list = np.array(coordinate_integrate.values.tolist())
    coordinate_matrix = pairwise_distances(coordinate_list, n_jobs=8)
    adj_matrix = np.where(coordinate_matrix >= distance_thresh, 0, 1)
    Edge_label = np.where(adj_matrix == 1)

    Adj_from = np.unique(Edge_label[0], return_counts=True)
    Adj_to = np.unique(Edge_label[1], return_counts=True)

    Adj_from_singleton = Adj_from[0][Adj_from[1] == 1]
    Adj_to_singleton = Adj_to[0][Adj_to[1] == 1]

    Adj_singleton = np.intersect1d(Adj_from_singleton, Adj_to_singleton)

    coordinate_matrix_modify = coordinate_matrix

    fromlist = Edge_label[0].tolist()
    tolist = Edge_label[1].tolist()

    edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
    graph.edge_index = edge_index
    
    connected_graph = g_util.to_networkx(graph, to_undirected=True)
    connected_graph = [connected_graph.subgraph(item_graph).copy() for item_graph in
                    nx.connected_components(connected_graph) if len(item_graph) > 50] # modify
    connected_graph_node_list = []
    for graph_item in connected_graph:
        connected_graph_node_list.extend(list(graph_item.nodes))
    connected_graph = connected_graph_node_list
    connected_graph = list(connected_graph)
    new_node_order_dict = dict(zip(connected_graph, range(len(connected_graph))))
    
    if connected_graph == []:
        print("connected_graph == []:", sample_name)
    else:
        new_feature = graph.x[connected_graph]
        new_edge_index = graph.edge_index.numpy()
        new_edge_mask_from = np.isin(new_edge_index[0], connected_graph)
        new_edge_mask_to = np.isin(new_edge_index[1], connected_graph)
        new_edge_mask = new_edge_mask_from * new_edge_mask_to
        new_edge_index_from = new_edge_index[0]
        new_edge_index_from = new_edge_index_from[new_edge_mask]
        new_edge_index_from = [new_node_order_dict[item] for item in new_edge_index_from]
        new_edge_index_to = new_edge_index[1]
        new_edge_index_to = new_edge_index_to[new_edge_mask]
        new_edge_index_to = [new_node_order_dict[item] for item in new_edge_index_to]

        new_edge_index = torch.tensor([new_edge_index_from, new_edge_index_to], dtype=torch.long)

        new_supernode = coordinate_integrate.iloc[connected_graph]
        new_supernode = new_supernode.reset_index()
        
        actual_pos = new_supernode[['X', 'Y']].to_numpy()
        actual_pos = torch.tensor(actual_pos)
        actual_pos = actual_pos.float()

        pos_transfrom = Polar()
        new_graph = Data(x=new_feature, edge_index=new_edge_index, pos=actual_pos * 200.0)
        new_graph = pos_transfrom(new_graph)
        
        torch.save(new_graph, os.path.join(save_path, str(distance_thresh) + "_artifact_sophis_final.pt"))
        
pool_obj = multiprocessing.Pool(64)
answer = pool_obj.map(supernode_graph_construct, sample_list)