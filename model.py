import torch
import torch.nn as nn
import torch.nn.functional as F

from locked_dropout import LockedDropout
from ON_LSTM import ONLSTMStack

class Pos_choser(nn.Module):
	### take in the tree currently generated, and return the distribution of positions to insert the next node
	def __init__(self, ntoken, hidden_dim, node_dim, tree_dim, dropout=0.1)
		super(Pos_choser,self).__init__()
		self.drop = nn.Dropout(dropout)
	###
		self.gcn = GCN(tree_dim)
	###
		self.inp_dim = hidden_dim + node_dim + tree_dim
		self.hidden_dim = hidden_dim
		self.node_dim = node_dim
		self.tree_dim = tree_dim
		self.score_cal = nn.Sequential(nn.Linear(self.inp_dim, self.hidden_dim), 
			nn.ReLU(),
			self.drop,
			nn.Linear(self.hidden_dim, 1))

	def forward(self, cur_tree)
		num_samples = cur_tree.size(0)
		cur_tree.make_index(0)
	###
		self.gcn(cur_tree)
		node_hidden = cur_tree.hidden_states ### should be a 2-D tensor
	###
		leaves = cur_tree.leaves(True)
		leave_inds = [x.index for x in leaves]
		leave_states = node_hidden[leave_inds]
		scores = map(self.score_cal.forward, leave_states)
		scores = F.softmax(scores)
		return leaves, scores

	def init_hidden(self):
		return self.score_cal.init_hidden()

class sentence_encoder(nn.Module):
	def __init__(self, ntoken, h_dim, emb_dim, nlayers, chunk_size, wdrop=0, dropouth=0.5):
		super(sentence_encoder, self).__init__()
		self.lockdrop = LockedDropout()
		self.hdrop = nn.Dropout(dropouth)
		self.encoder = nn.Embedding(ntoken, emb_dim)
		self.rnn = ONLSTMStack(
			[emb_dim] + [h_dim] + nlayers,
			chunk_size = chunk_size,
			dropconnect = wdrop,
			dropout = dropouth
		)
		initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.chunk_size = chunk_size
        self.wdrop = wdrop
        self.dropouth = dropouth

    def forward(self, inp_sentence, hidden)


class word_choser(nn.Module):
	def __init__(self, ntoken, hidden_dim):
