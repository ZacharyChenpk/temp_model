import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from locked_dropout import LockedDropout
from ON_LSTM import ONLSTMStack
from fakegcn import Graph

class Pos_choser(nn.Module):
	### Take in the tree currently generated, and return the distribution of positions to insert the next node
	def __init__(self, ntoken, node_dim, dropout=0.1):
		super(Pos_choser,self).__init__()
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, node_dim)
	###
	#	self.gcn = GCN()
	#	self.aggregation = pool()
	###
		self.inp_dim = node_dim * 2
		self.node_dim = node_dim
		'''
			The score_cal network will take in the GCN result of a position and the aggregation result of the whole graph,
			then calculate the score of choosing this position
		'''
		self.score_cal = nn.Sequential(nn.Linear(self.inp_dim, self.node_dim), 
			nn.ReLU(),
			self.drop,
			nn.Linear(self.node_dim, 1))

	def forward(self, cur_tree, sentence_encoder, dictionary):
		num_samples = cur_tree.nodenum()
		cur_tree.make_index(0)
		###
		'''
		self.gcn(cur_tree)
		node_hidden = cur_tree.hidden_states ### should be a 2-D tensor
		graph_hidden = self.aggregation(cur_tree) 
		graph_hidden = graph_hidden.repeat(num_samples)
		node_hidden = torch.cat((node_hidden, graph_hidden), 1)
		'''
		the_graph = cur_tree.tree2graph(self, dictionary, self.node_dim)
		node_hidden = the_graph.node_embs
		graph_hidden = the_graph.the_aggr()
		#print('graph_hidden size:', graph_hidden.size())
		graph_hidden = graph_hidden.repeat(num_samples).view(-1,self.node_dim)
		node_hidden = torch.cat((node_hidden, graph_hidden), 1)
		###
		leaves = cur_tree.leaves(True)
		leave_inds = [x.index for x in leaves]
		leave_states = node_hidden[leave_inds]
		scores = list(map(self.score_cal, leave_states))
		scores = F.softmax(torch.Tensor(scores))
		### Return available positions, their indexes, and their distribution of probability
		return leaves, leave_inds, scores

	def init_hidden(self):
		for layer in self.score_cal:
			if isinstance(layer, nn.Linear):
				torch.nn.init.xavier_uniform_(layer.weight)

class sentence_encoder(nn.Module):
	### take in a sentence, return its encoded embedding and hidden states(for attention)
	def __init__(self, ntoken, h_dim, emb_dim, nlayers, chunk_size, wdrop=0, dropouth=0.5):
		super(sentence_encoder, self).__init__()
		self.lockdrop = LockedDropout()
		self.hdrop = nn.Dropout(dropouth)
		self.encoder = nn.Embedding(ntoken, emb_dim)
		self.rnn = nn.LSTM(emb_dim, h_dim, nlayers)
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.h_dim = h_dim
		self.emb_dim = emb_dim
		self.nlayers = nlayers
		self.ntoken = ntoken
		self.chunk_size = chunk_size
		self.wdrop = wdrop
		self.dropouth = dropouth

	def forward(self, inp_sentence, hidden):
		emb = list(map(lambda x:self.encoder(torch.LongTensor([x])).squeeze(), inp_sentence))
		#print('inp sen: ', inp_sentence)
		#print('emb: ', emb)
		h0 = torch.randn(self.nlayers, 1, self.h_dim)
		c0 = torch.randn(self.nlayers, 1, self.h_dim)
		output = []
		h_n = torch.zeros(len(inp_sentence), self.h_dim)
		for i in range(len(emb)):
			opt,(h,c)=self.rnn(torch.Tensor(emb[i]).unsqueeze(1), (h0,c0))
			output.append(opt)
			h_n[i]=h.squeeze()[-1]

		#print('hn:', h_n)

		# output is the hidden states, and h_n is the encoding result
		return output, h_n

	def init_hidden(self, bsz):
		return 0

class naiveLSTMCell(nn.Module):
	### In our model, we have to involve per step of LSTM with tree processing and attention, so we rewrite the single cell of LSTM to deal with it
	### Hope for successful parallel computing!
	def __init__(self, inp_size, hidden_size):
		super(naiveLSTMCell, self).__init__()
		self.inp_size = inp_size
		self.hidden_size = hidden_size

		self.inp_i = nn.Linear(inp_size, hidden_size)
		self.inp_h = nn.Linear(hidden_size, hidden_size)
		self.forget_i = nn.Linear(inp_size, hidden_size)
		self.forget_h = nn.Linear(hidden_size, hidden_size)
		self.out_i = nn.Linear(inp_size, hidden_size)
		self.out_h = nn.Linear(hidden_size, hidden_size)
		self.cell_i = nn.Linear(inp_size, hidden_size)
		self.cell_h = nn.Linear(hidden_size, hidden_size)

		self.init_weights()
		self.cur_cell = torch.zeros(hidden_size)
		self.cur_h = torch.zeros(hidden_size)

	def init_weights(self):
		stdv = 1. / math.sqrt(self.hidden_size)
		self.inp_i.bias.data.fill_(0)
		self.inp_i.weight.data.uniform_(-stdv, stdv)
		self.inp_h.bias.data.fill_(0)
		self.inp_h.weight.data.uniform_(-stdv, stdv)
		self.forget_i.bias.data.fill_(0)
		self.forget_i.weight.data.uniform_(-stdv, stdv)
		self.forget_h.bias.data.fill_(0)
		self.forget_h.weight.data.uniform_(-stdv, stdv)
		self.out_i.bias.data.fill_(0)
		self.out_i.weight.data.uniform_(-stdv, stdv)
		self.out_h.bias.data.fill_(0)
		self.out_h.weight.data.uniform_(-stdv, stdv)
		self.cell_i.bias.data.fill_(0)
		self.cell_i.weight.data.uniform_(-stdv, stdv)
		self.cell_h.bias.data.fill_(0)
		self.cell_h.weight.data.uniform_(-stdv, stdv)

	def init_cellandh(self):
		'''
		stdv = 1. / math.sqrt(self.hidden_size)
		self.cur_cell.data.uniform_(-stdv, stdv)
		self.cur_h.data.uniform_(-stdv, stdv)
		'''
		self.cur_cell.data.fill_(0)
		self.cur_h.data.fill_(0)

	def forward(self, inp):
		i = torch.sigmoid(self.inp_i(inp) + self.inp_h(self.cur_h))
		f = torch.sigmoid(self.forget_i(inp) + self.forget_h(self.cur_h))
		g = torch.tanh(self.cell_i(inp) + self.cell_h(self.cur_h))
		o = torch.sigmoid(self.out_i(inp) + self.out_h(self.cur_h))
		self.cur_cell = f * self.cur_cell + i * g
		self.cur_h = o * torch.tanh(self.cur_cell)
		return self.cur_cell, self.cur_h

class word_choser(nn.Module):
	def __init__(self, ntoken, ntoken_out, hidden_dim, emb_dim, chunk_size, nlayers):
		super(word_choser, self).__init__()
		self.lockdrop = LockedDropout()
		self.dim_up = torch.nn.Parameter(torch.zeros((hidden_dim, ntoken), requires_grad=True))
		self.dim_down = torch.nn.Parameter(torch.zeros((ntoken, hidden_dim),requires_grad=True))
		self.dim_out = torch.nn.Parameter(torch.zeros((hidden_dim, ntoken_out),requires_grad=True))
		self.inpdim = emb_dim + hidden_dim + 1
		self.outdim = hidden_dim
		###
		#	self.attention_gcn = GCN()
		#	self.attention_pool = pool()
		###
		self.ntoken = ntoken
		self.ntoken_out = ntoken_out
		self.hidden_dim = hidden_dim
		self.emb_dim = emb_dim
		self.chunk_size = chunk_size
		self.nlayers = nlayers
		self.lstm = naiveLSTMCell(self.inpdim, hidden_dim)
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.dim_up.data.uniform_(-initrange, initrange)
		self.dim_down.data.uniform_(-initrange, initrange)
		self.dim_out.data.uniform_(-initrange, initrange)
		self.lstm.init_weights()
		self.lstm.init_cellandh()

	def forward(self, sen_emb, hiddens, pos_index):
		#print(hiddens.size())
		#print(sen_emb.size())
		att_scores = hiddens.mm(sen_emb.view(-1,1))
		att_scores = torch.sigmoid(att_scores)
		att_scores = F.softmax(att_scores)
		att_result = att_scores.t().mm(hiddens)
		#print('att_result:',att_result)

		the_inp = torch.cat((sen_emb, att_result.squeeze(0), torch.Tensor([pos_index])))
		#print('the_inp:',the_inp)
		_, h = self.lstm(the_inp)
		#print(h.size())
		#print(self.dim_out.size())
		h = h.unsqueeze(0).mm(self.dim_out).squeeze(0)
		h = F.softmax(h)
		#print(h.size())
		return h

if __name__ == "__main__":
	pc = Pos_choser(1, 1)
	se = sentence_encoder(1, 1, 1, 1, 1)
	nlc = naiveLSTMCell(1, 1)
	wc = word_choser(1, 1, 1, 1, 1)
