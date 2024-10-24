# coding=utf-8
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing, GATConv, GATv2Conv

from ..common import MeanReadout, SumReadout, MultiLayerPerceptron      


class GATEncoder(torch.nn.Module):

    def __init__(self, hidden_dim, num_convs=4, activation='relu', short_cut=True, concat_hidden=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(GATv2Conv(128, 16, 8, True))
        # self.convs.append(GATConv(128, 128, 16))

    

    def forward(self, z, edge_index):
        """
        Input:
            data: (torch_geometric.data.Data): batched graph
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
        Output:
            node_attr
            graph feature
        """

 
        hiddens = []
        conv_input = z # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden = conv_input+ hidden

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return node_feature
