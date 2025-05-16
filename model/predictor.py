
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from .GraphTransformer import GraphTransformer
from .common import NodeFeatures
import os




class Predictor(torch.nn.Module):
    def __init__(self, max_layer = 6, num_features_drug = 78, num_nodes = 200, num_relations_mol = 10, num_relations_graph = 10, output_dim=64, max_degree_graph=100, max_degree_node=100, sub_coeff = 0.2, mi_coeff = 0.5, dropout=0.2, device = 'cuda', args=None):
        super(Predictor, self).__init__()

        self.device = device

        self.layers = max_layer
        self.num_features_drug = num_features_drug

        self.max_degree_graph = max_degree_graph
        self.max_degree_node = max_degree_node

        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff
        self.dropout = dropout

        self.mol_atom_feature = NodeFeatures(degree=max_degree_graph, feature_num=num_features_drug, embedding_dim=output_dim, type='graph')
        self.drug_node_feature = NodeFeatures(degree=max_degree_node, feature_num=num_nodes, embedding_dim=output_dim, type='node')

        self.mol_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_mol, dropout= dropout, type='graph')
        self.node_representation_learning = GraphTransformer(layer_num = max_layer, embedding_dim = output_dim, num_heads = 4, num_rel = num_relations_graph, dropout=dropout, type='node')

        self.fc1 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

        self.pos = args.pos
        self.neg = args.neg

    def to(self, device):

        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)

        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)

        self.fc1.to(device)
        self.fc2.to(device)

        self.disc.to(device)
        self.b_xent.to(device)

    def reset_parameters(self):

        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()

        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()


    def forward(self, drug1_mol, drug2_mol, subgraph):

        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        drug_node_feature = self.drug_node_feature(subgraph)

        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        drug1_node_embedding, drug2_node_embedding = self.node_representation_learning(drug_node_feature, subgraph)


        drug1_embedding = self.fc1(torch.concat([drug1_node_embedding, mol1_graph_embedding],dim=-1))
        drug2_embedding = self.fc1(torch.concat([drug2_node_embedding, mol2_graph_embedding],dim=-1))

        score = self.fc2(torch.concat([drug1_embedding, drug2_embedding], dim=-1))

        loss_s_m = self.loss_MI(self.MI(drug1_embedding, mol1_atom_embedding)) + self.loss_MI(self.MI(drug2_embedding, mol2_atom_embedding))
        # loss_s_d = self.loss_MI(self.MI(drug1_embedding, drug1_sub_embedding)) + self.loss_MI(self.MI(drug2_embedding, drug2_sub_embedding))


        predicts_drug = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(predicts_drug, drug1_mol.y.view(-1))

        # loss = loss_label + self.mol_coeff* loss_s_m + self.mi_coeff * loss_s_d
        loss = loss_label + self.mol_coeff* loss_s_m

        return torch.exp(predicts_drug)[:,1], loss

    def pred(self, drug1_mol, drug2_mol, subgraph_lists, batch):

        mol1_atom_feature = self.mol_atom_feature(drug1_mol)
        mol2_atom_feature = self.mol_atom_feature(drug2_mol)

        mol1_graph_embedding, mol1_atom_embedding, mol1_attn = self.mol_representation_learning(mol1_atom_feature, drug1_mol)
        mol2_graph_embedding, mol2_atom_embedding, mol2_attn = self.mol_representation_learning(mol2_atom_feature, drug2_mol)

        # mol1_graph_embedding batch_size * dim

        drug_node_feature = self.drug_node_feature(subgraph_lists)
        drug1_node_embedding, drug2_node_embedding = self.node_representation_learning(drug_node_feature, subgraph_lists)
        # print('drug node feature:', drug_node_feature)
        # print('drug1_node_embedding:', drug1_node_embedding)
        # print('drug2_node_embedding:', drug2_node_embedding)

        drug1_embedding = self.fc1(torch.concat([drug1_node_embedding, mol1_graph_embedding[batch]],dim=-1))
        drug2_embedding = self.fc1(torch.concat([drug2_node_embedding, mol2_graph_embedding[batch]],dim=-1))

        score = self.fc2(torch.concat([drug1_embedding, drug2_embedding], dim=-1))
        # print('score:', score)
        predicts_drug = F.log_softmax(score, dim=-1)
        predicts_drug = torch.exp(predicts_drug)[:,1]
        # print('pred', predicts_drug)

        return predicts_drug
    
    def get_reward(self, drug1_mol, drug2_mol, subgraph_lists, batch, pred_default=None):


        predicts_drug = self.pred(drug1_mol, drug2_mol, subgraph_lists, batch)
        y = drug1_mol.y.view(-1)[batch]
        

        """How to design reward function?"""
        # v1
        # condition1 = (y == 1) & (predicts_drug > 0.5)
        # condition2 = (y == 0) & (predicts_drug <= 0.5)
        # reward = torch.where(condition1 | condition2, 2, -2)

        # v2
        # pred_default = pred_default[batch]
        # reward_1 = 5*(predicts_drug - pred_default)
        # reward_0 = 5*(pred_default - predicts_drug)
        # reward = torch.where(y == 1, reward_1, reward_0)

        # v3
        reward = torch.zeros_like(predicts_drug)
        pred_default = pred_default[batch]

        cond1 = (y == 1) & (predicts_drug > pred_default)
        reward[cond1] = self.pos * (predicts_drug[cond1] - pred_default[cond1])

        cond2 = (y == 1) & (predicts_drug <= pred_default)
        reward[cond2] = self.neg * (predicts_drug[cond2] - pred_default[cond2])

        cond3 = (y == 0) & (predicts_drug < pred_default)
        reward[cond3] = self.pos * (pred_default[cond3] - predicts_drug[cond3])

        cond4 = (y == 0) & (predicts_drug >= pred_default)
        reward[cond4] = self.neg * (pred_default[cond4] - predicts_drug[cond4])

        return reward, predicts_drug


    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub)) ##pos
            c_1_list.append(c_1.expand_as(sub)) ##neg
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):

        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)

        return self.b_xent(logits.view([1,-1]), lbl.view([1, -1]))

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__+'.pt')
        torch.save(self.state_dict(), save_path)
        return save_path



class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: 1, 512; h_pl: 1, 2708, 512; h_mi: 1, 2708, 512
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits



def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)