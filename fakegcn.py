import os
import torch
import sys
import numpy as np

from collections import Counter
import nltk

### This file is just used for test :)

class Graph(object):
	def __init__(self, nodenum, nodedim, edgedim):
		self.nodenum = nodenum
		self.nodedim = nodedim
		self.edgedim = edgedim
		self.node_embs = torch.zeros(nodenum, nodedim)
		self.adj = torch.zeros(nodenum, nodenum)
		self.edge_embs = torch.zeros(nodenum, nodenum, edgedim)
		self.node_names = [''] * nodenum

	def match_tree(self, tree, sen_encoder, dictionary):
		embeder = sen_encoder.encoder
		word_id = dictionary.word2idx[tree.root]
		#word_dis = torch.zeros(len(dictionary.idx2word), dtype = torch.long)
		#print(len(dictionary.idx2word))
		#word_dis[word_id] = 1
		the_emb = embeder(torch.LongTensor([word_id]))
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

	def the_gcn(self):
		dim_up = torch.normal(torch.zeros(self.nodedim, self.edgedim), torch.ones(self.nodedim, self.edgedim), out=None)
		for i in range(self.nodenum):
			for j in range(self.nodenum):
				self.edge_embs[i][j] = self.adj * self.node_embs[i].mul(self.node_embs[j])

	def the_aggr(self):
		return torch.squeeze(torch.mean(self.node_embs, dim=0),0)
