import math
import numpy as np
import torch.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class Graph(nn.Module):
    def __init__(self, g, nfeat, nhid, nclass, dropout):
        super(Graph, self).__init__()
        
        n = g.shape[0]
        adj = torch.zeros([n, n])
        for i in range(n):
            for j in range(i):
                mv = torch.max(g[j:i])
                dis = mv - g[i] + mv - g[j]
                adj[i][j] = dis
                adj[j][i] = dis
        adjt = adj.transpose(0, 1)
        c = (adjt > adj).float()
        adj = adj + adjt.mul(c) - adj.mul(c)
        adj = adj + torch.eye(adj.shape[0])
        self.gcn = GCN(nfeat, nhid, nclass, dropout)
        self.adj = adj
    
    def forward(self, x):
        return self.gcn(x, self.adj)

if __name__ == "__main__":
    g = torch.Tensor([1, 1, 1, 2, 3, 4])
    model = Graph(g, 10, 3, 2, 0.5)
