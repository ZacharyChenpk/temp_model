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
    def __init__(self, nfeature, nhidden, noutput, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeature, nhidden)
        self.gc2 = GraphConvolution(nhidden, noutput)
        self.dropout = dropout

    def forward(self, x, adj):
        adjt = adj.transpose(0, 1)
        c = (adjt > adj).float()
        adj = adj + adjt.mul(c) - adj.mul(c)
        adj = adj + torch.eye(adj.shape[0])

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
'''
class Graph(nn.Module):
    def __init__(self, g, nfeature, nhidden, noutput, dropout):
        super(Graph, self).__init__()
        
        n = g.shape[0]
        adj = torch.zeros([n, n])
        for i in range(n):
            for j in range(i):
                mv = torch.max(g[j:i])
                dis = mv - g[i] + mv - g[j]
                adj[i][j] = dis
                adj[j][i] = dis
        
        self.gcn = GCN(nfeature, nhidden, noutput, dropout)
        self.adj = adj
    
    def forward(self, x):
        return self.gcn(x, self.adj)
'''
class Graph(object):
    def __init__(self, nodenum, nodedim, edgedim):
        self.nodenum = nodenum
        self.nodedim = nodedim
        self.edgedim = edgedim
        self.node_embs = torch.zeros(nodenum, nodedim)
        self.adj = torch.zeros(nodenum, nodenum)
        self.edge_embs = torch.zeros(nodenum, nodenum, edgedim)
        self.node_names = [''] * nodenum

    def hiddenG_init(self):
        for i in range(self.nodenum):
            for j in range(i):
                mv = torch.max(g[j:i])
                dis = mv - g[i] + mv - g[j]
                self.adj[i][j] = dis
                self.adj[j][i] = dis

    def match_tree(self, tree, sen_encoder, dictionary):
        #embeder = sen_encoder.encoder
        word_id = dictionary.word2idx[tree.root]
        #word_dis = torch.zeros(len(dictionary.idx2word), dtype = torch.long)
        #print(len(dictionary.idx2word))
        #word_dis[word_id] = 1
        the_emb = sen_encoder(torch.LongTensor([word_id]))
        #print('node_embs size:', self.node_embs.size())
        #print('the_emb size:', the_emb.size())
        self.node_embs[tree.index] = the_emb
        self.node_names[tree.index] = tree.root
        if tree.left:
            self.adj[tree.index][tree.left.index] = 1
            self.adj[tree.left.index][tree.index] = 1
            self.match_tree(tree.left, sen_encoder, dictionary)
        if tree.right:
            self.adj[tree.index][tree.right.index] = 1
            self.adj[tree.right.index][tree.index] = 1
            self.match_tree(tree.right, sen_encoder, dictionary)

    def ram_full_init(self):
        self.adj.fill_(0)
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def the_gcn(self, gcn):
        self.node_embs = gcn(self.node_embs, self.adj)
        return self.node_embs

    def the_aggr(self):
        return torch.squeeze(torch.mean(self.node_embs, dim=0),0)
    