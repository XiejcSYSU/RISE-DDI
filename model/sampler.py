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
import os



class Sampler(torch.nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()

        self.args = args

        pass

    def forward(self):
        pass

    def predict(self):
        pass