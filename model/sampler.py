import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree, subgraph, k_hop_subgraph
from torch_geometric.utils import *
from .GraphTransformer import GraphTransformer
import os
from torch_scatter import scatter_max, scatter_softmax
from .common import NodeFeatures
import torch_geometric as geometric
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from joblib import Parallel, delayed



class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    Input: embedding matrix for knowledge graph entity and adjacency matrix
    Output: gcn embedding for kg entity
    """

    def __init__(self, in_channel=[64,32], out_channel=[32,64]):
        super(GraphConv, self).__init__()

        self.conv1 = geometric.nn.SAGEConv(in_channel[0], out_channel[0])
        self.conv2 = geometric.nn.SAGEConv(in_channel[1], out_channel[1])

    def forward(self, x, edge_indices):
        x = self.conv1(x, edge_indices)
        x = F.leaky_relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_indices)
        x = F.dropout(x)
        x = F.normalize(x)

        return x

class Prior(torch.nn.Module):
    def __init__(self, args, state_dim, action_dim):
        super(Prior, self).__init__()
        
        self.fc = [
                   nn.Linear(state_dim, 32),
                   nn.LayerNorm(32),
                   nn.Tanh(),
                   nn.Linear(32, action_dim)]
        # self.fc = [nn.Linear(state_dim, action_dim)]
        self.fc_layers = nn.Sequential(*self.fc)

        self.fc1 = nn.Linear(2*state_dim, state_dim)


        self.conv1d = nn.Conv1d(
            in_channels=2,    # 输入通道数 (对应 2xN 中的 2)
            out_channels=1,   # 输出通道数 (卷积核的数量)
            kernel_size=3,    # 卷积核的大小
            stride=1,         # 步长
            padding=1         # 填充，让输出长度和输入一致
        )

    
    def forward(self, e1, e2, batch=None):

        # e = torch.stack((e1, e2), dim=1)  # batch_size * 2 * d_dim
        # e = self.conv1d(e)  # batch_size * 1 * d_dim
        # e = e.squeeze(1)
        # e = F.relu(e)

        e2 = self.fc1(torch.cat((e1, e2), dim=-1))

        e = e1 + e2

        logits = self.fc_layers(e).squeeze(1)
        if batch is not None:
            logits = scatter_softmax(logits, batch)
        else:
            logits = F.softmax(logits, dim=0)

        return logits


class Sampler(torch.nn.Module):
    def __init__(self, args, num_nodes):
        super(Sampler, self).__init__()

        self.args = args
        self.prior = Prior(args, args.d_dim, 1)
        self.num_nodes = num_nodes

        self.done_embedding = nn.Embedding(1, args.d_dim)
        self.gcn = GraphConv(in_channel=[args.d_dim, args.d_dim//2], out_channel=[args.d_dim//2, args.d_dim])


    def forward(self, data_idx, adj_matrix, edge_rel, embeddings, origin_graph):


        x = embeddings
        # graph_embeddings = self.gcn(x, adj_matrix)
        graph_embeddings = x

        graph_embeddings = torch.concat((graph_embeddings, self.done_embedding.weight), dim=0) 

        batch_size = data_idx.shape[0]
        subgraph_list = []
        batch = []
        prob_list = torch.tensor([], device=adj_matrix.device)

        tmp = []


        k = self.args.k_step
        num_sample = 1

        origin_graphs = origin_graph.to_data_list()

        """sample candidate subgraph baed on knowledge graph"""
        for idx, node_pair in enumerate(data_idx):
            current_nodes = node_pair
            subset, _, _, _ = k_hop_subgraph(
                node_idx=node_pair,  
                num_hops=1,         
                edge_index=adj_matrix,  
                relabel_nodes=False  
            )
            for _ in range(k):
                batch.append(idx)
                """action space"""
                neighbors = self.get_neighbors(current_nodes, adj_matrix, subset, origin_graphs[idx])

                """prior network"""
                neighbors_embeddings =  graph_embeddings[neighbors] # n neighbors
                k_embeddings = graph_embeddings[current_nodes].mean(dim=0)  # m nodes of subgraph -> mean -> 1

                logits = self.prior(k_embeddings.expand_as(neighbors_embeddings), neighbors_embeddings)

                """sampled neighbors"""
                sampled_indices = torch.multinomial(logits, num_sample, replacement=True)
                
                if len(logits)-1 in sampled_indices:
                    """done"""
                    sampled_logits = logits[sampled_indices].mean()
                    sampled_indices = sampled_indices[sampled_indices != len(logits)-1]
                    sampled_neighbors = neighbors[sampled_indices]
                    current_nodes = torch.concat((current_nodes, sampled_neighbors), dim=-1)  
                    subgraph_list.append(self.generate_subgraph(current_nodes, adj_matrix, edge_rel, origin_graphs[idx]))
                    prob_list = torch.cat([prob_list, sampled_logits.unsqueeze(0)])
                    tmp.append((idx, -1))
                    break
                else:
                    sampled_neighbors = neighbors[sampled_indices]
                    sampled_logits = logits[sampled_indices].mean()
                    current_nodes = torch.concat((current_nodes, sampled_neighbors), dim=-1)  

                    subgraph_list.append(self.generate_subgraph(current_nodes, adj_matrix, edge_rel, origin_graphs[idx]))
                    prob_list = torch.cat([prob_list, sampled_logits.unsqueeze(0)])
                    tmp.append((idx, sampled_neighbors.item()))
                    

        subgraph_list = Batch.from_data_list(subgraph_list)
        batch = torch.tensor(batch)

        return subgraph_list, prob_list, batch, tmp

        #########################################################################################################
        """sample candidate subgraph baed on knowledge graph"""

        subset_list = []
        for idx, node_pair in enumerate(data_idx):
            subset, _, _, _ = k_hop_subgraph(
                node_idx=node_pair,  
                num_hops=1,         
                edge_index=adj_matrix,  
                relabel_nodes=False  
            )
            subset_list.append(subset)

        current_nodes = data_idx

        done = torch.zeros(batch_size, dtype=torch.bool).cuda()
        from time import time
        for _ in range(k):

            """action space"""
            neighbors, in_batch = self.get_neighbors_batch(current_nodes, adj_matrix, subset_list)
            
            """prior network"""
            neighbors_embeddings =  graph_embeddings[neighbors] # n neighbors

            k_embeddings = graph_embeddings[current_nodes].mean(dim=1) 
            # all_embeddings = torch.cat((k_embeddings[in_batch], neighbors_embeddings), dim=-1)
            logits = self.prior(k_embeddings[in_batch], neighbors_embeddings, in_batch)

            """sampled neighbors"""
            sampled_indices = self.sample(logits, in_batch)

            sampled_neighbors = neighbors[sampled_indices] # batch_size * 1
            sampled_logits = logits[sampled_indices] 
            current_nodes = torch.concat((current_nodes, sampled_neighbors.unsqueeze(-1)), dim=-1) # batch_size * n

            current_nodes = current_nodes[done == False]
            sampled_logits = sampled_logits[done == False]
            idx = torch.arange(batch_size).cuda()[done == False]

            done = sampled_neighbors == self.num_nodes

            for i in range(current_nodes.shape[0]):
                subgraph_list.append(self.generate_subgraph(current_nodes[i], adj_matrix, edge_rel))
                batch.append(idx[i])
            prob_list = torch.cat([prob_list, sampled_logits], dim=0) 


        subgraph_list = Batch.from_data_list(subgraph_list)
        batch = torch.tensor(batch)
        
        return subgraph_list, prob_list, batch, tmp




    
    def predict(self, data_idx, adj_matrix, edge_rel, embeddings, origin_graph):
        with torch.no_grad():
            x = embeddings
            # graph_embeddings = self.gcn(x, adj_matrix)
            graph_embeddings = x
            graph_embeddings = torch.concat((graph_embeddings, self.done_embedding.weight), dim=0) 

            k = self.args.k_step
            num_sample = 1

            subgraph_list = []
            batch = []

            origin_graphs = origin_graph.to_data_list()

            for idx, node_pair in enumerate(data_idx):
                current_nodes = node_pair
                subset, _, _, _ = k_hop_subgraph(
                    node_idx=node_pair,  
                    num_hops=1,         
                    edge_index=adj_matrix,  
                    relabel_nodes=False  
                )
                for i in range(k):

                    neighbors = self.get_neighbors(current_nodes, adj_matrix, subset, origin_graphs[idx]) 

                    neighbors_embeddings =  graph_embeddings[neighbors] # n neighbors
                    k_embeddings = graph_embeddings[current_nodes].mean(dim=0)  # m nodes of subgraph -> mean -> 1

                    logits = self.prior(k_embeddings.expand_as(neighbors_embeddings), neighbors_embeddings)

                    sampled_indices = torch.multinomial(logits, num_sample, replacement=True) 

                    if len(logits)-1 in sampled_indices:
                        # sampled_indices = sampled_indices[sampled_indices != len(logits)-1]
                        # sampled_neighbors = neighbors[sampled_indices]
                        # current_nodes = torch.concat((current_nodes, sampled_neighbors), dim=-1)  
                        break
                    else:
                        sampled_neighbors = neighbors[sampled_indices]
                        current_nodes = torch.concat((current_nodes, sampled_neighbors), dim=-1) 

                subgraph_list.append(self.generate_subgraph(current_nodes, adj_matrix, edge_rel, origin_graphs[idx]))
                batch.append(idx)

            subgraph_list = Batch.from_data_list(subgraph_list)
            batch = torch.tensor(batch)
            return subgraph_list, batch



    def generate_default_subgraph(self, data_idx, adj_matrix, edge_rel):
        subgraph_list = []
        batch = []

        for idx, node_pair in enumerate(data_idx):
            current_nodes = node_pair
            subgraph_list.append(self.generate_subgraph(current_nodes, adj_matrix, edge_rel))
            batch.append(idx)

        subgraph_list = Batch.from_data_list(subgraph_list)
        batch = torch.tensor(batch)

        return subgraph_list, batch

        

    def sample(self, logits, batch):
        unique_batches = batch.unique()
        sampled_indices = []
        for b in unique_batches:
            mask = batch == b
            group_probs = logits[mask]
            group_indices = torch.nonzero(mask, as_tuple=True)[0]

            sampled = torch.multinomial(group_probs, num_samples=1, replacement=True)  
            sampled_indices.append(group_indices[sampled])  

        sampled_indices = torch.cat(sampled_indices)
        return sampled_indices
    
    def get_neighbors(self, current_nodes, adj_matrix, subset, origin_graph): # not batch

        x = origin_graph.x

        src, dst = adj_matrix  
        mask = torch.isin(src, current_nodes)
        neighbors = dst[mask]       
        neighbors = torch.cat((neighbors, torch.tensor([self.num_nodes]).cuda()))   # add terminate node

        mask = torch.isin(neighbors, subset)
        neighbors = neighbors[mask]
        mask = ~torch.isin(neighbors, x)
        neighbors = neighbors[mask]
        mask = ~torch.isin(neighbors, current_nodes)
        neighbors = neighbors[mask]

        return torch.unique(neighbors).cuda()
    
    def get_neighbors_batch(self, current_nodes, adj_matrix, subset):
        batch = []
        neighbors_list = []

        current_nodes2 = current_nodes.cpu()
        adj_matrix2 = adj_matrix.cpu()
        subset2 = subset

        for i in range(current_nodes.shape[0]):

            src, dst = adj_matrix2
            mask = torch.isin(src, current_nodes2[i])
            neighbors = dst[mask]    
            neighbors = torch.cat((neighbors, torch.tensor([self.num_nodes])))

            mask = torch.isin(neighbors, subset[i].cpu())
            neighbors = neighbors[mask]
            mask = ~torch.isin(neighbors, current_nodes2[i])
            neighbors = neighbors[mask]

            neighbors = torch.unique(neighbors)  

            neighbors_list.append(neighbors)
            batch.append(torch.full((len(neighbors),), i))

        neighbors_list = torch.cat(neighbors_list).cuda()
        batch = torch.cat(batch).cuda()
        return neighbors_list, batch

    
    def generate_subgraph(self, current_nodes, adj_matrix, edge_rels, origin_graph):
        nodes = origin_graph.x
        nodes_id = current_nodes[:2]
        current_nodes = torch.cat((current_nodes, nodes), dim=0)
        current_nodes = torch.unique(current_nodes)

        mask = (current_nodes != nodes_id[0]) & (current_nodes != nodes_id[1])
        current_nodes = current_nodes[mask]
        current_nodes = torch.cat([nodes_id, current_nodes])

        mapping_id = torch.zeros(len(current_nodes), dtype=torch.long)
        mapping_id[torch.where(current_nodes == nodes_id[0])[0]] = 1
        mapping_id[torch.where(current_nodes == nodes_id[1])[0]] = 1
        # mapping_id[:2] = 1

        edge_index, edge_rel = subgraph(current_nodes, adj_matrix, edge_rels, relabel_nodes=True)
        x = current_nodes.cpu()
        
        G_sub = DATA.Data(x=x,
                  edge_index=edge_index,
                  id=mapping_id,
                  rel_index=edge_rel,
                  sp_edge_index=edge_index,
                  sp_value=torch.ones(edge_index.size(1), dtype=torch.float),
                  sp_edge_rel=edge_rel
                )
        
        return G_sub
