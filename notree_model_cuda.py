import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from locked_dropout import LockedDropout
from ON_LSTM import ONLSTMStack
from GCN import Graph, GCN
from tree import Tree

class Pos_choser(nn.Module):
    ### Take in the tree currently generated, and return the distribution of positions to insert the next node
    def __init__(self, ntoken, node_dim, emb_dim, ntoken_out, dropout=0.1):
        super(Pos_choser,self).__init__()
        self.drop = nn.Dropout(dropout).cuda()
        self.gcn = GCN(node_dim, node_dim, node_dim, dropout, cuda=True).cuda()
        self.ntoken_out = int(ntoken_out)
    ###
    #   self.gcn = GCN()
    #   self.aggregation = pool()
    ###
        self.inp_dim = node_dim * 2 + emb_dim
        self.emb_dim = emb_dim
        self.node_dim = node_dim
        '''
            The score_cal network will take in the GCN result of a position and the aggregation result of the whole graph,
            then calculate the score of choosing this position
        '''
        self.score_cal = nn.Sequential(nn.Linear(self.inp_dim, self.node_dim), 
            nn.ReLU(),
            self.drop,
            nn.Linear(self.node_dim, 1))

    def forward(self, cur_tree, chosen_word, sentence_encoder, dictionary):
        num_samples = cur_tree.nodenum()
        cur_tree.make_index(0)
        #chosen_wordemb = torch.zeros(self.ntoken_out, requires_grad=False, dtype = torch.long)
        #chosen_wordemb[dictionary.word2idx[chosen_word]] = 1
        #chosen_wordemb = sentence_encoder.encoder(chosen_wordemb)
        chosen_wordemb = sentence_encoder(torch.LongTensor([dictionary.word2idx[chosen_word]]).cuda()).squeeze(0)
        chosen_wordemb.requeires_grad=False

        ###
        '''
        self.gcn(cur_tree)
        node_hidden = cur_tree.hidden_states ### should be a 2-D tensor
        graph_hidden = self.aggregation(cur_tree) 
        graph_hidden = graph_hidden.repeat(num_samples)
        node_hidden = torch.cat((node_hidden, graph_hidden), 1)
        '''
        the_graph = cur_tree.tree2graph(sentence_encoder, dictionary, self.node_dim, cuda=True)
        the_graph.the_gcn(self.gcn)
        node_hidden = the_graph.node_embs
        graph_hidden = the_graph.the_aggr()
        graph_hidden = graph_hidden.repeat(num_samples).view(-1, self.node_dim)
        word_embs = chosen_wordemb.repeat(num_samples).view(-1, self.emb_dim)
        node_hidden = torch.cat((node_hidden, graph_hidden), 1)
        node_hidden = torch.cat((word_embs, node_hidden), 1)
        ###
        leaves = cur_tree.leaves(True)
        leave_inds = [x.index for x in leaves]
        leave_states = node_hidden[leave_inds]
        scores = self.score_cal(leave_states)
        scores = scores.view(-1)
        scores = F.softmax(scores)
        ### Return available positions, their indexes, and their distribution of probability
        return leaves, leave_inds, scores

    def init_hidden(self):
        for layer in self.score_cal:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

class sentence_encoder(nn.Module):
    ### take in a sentence, return its encoded embedding and hidden states(for attention)
    def __init__(self, ntoken, h_dim, emb_dim, nlayers, chunk_size, init_emb, wdrop=0, dropouth=0.5):
        super(sentence_encoder, self).__init__()
        #self.lockdrop = LockedDropout()
        #self.hdrop = nn.Dropout(dropouth)
        self.encoder = nn.Embedding.from_pretrained(init_emb).cuda()
        self.rnn = nn.LSTM(emb_dim, h_dim, nlayers)
        self.rnn = self.rnn.cuda()
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.chunk_size = chunk_size
        self.wdrop = wdrop
        self.dropouth = dropouth

    def forward(self, inp_sentence):
        #print(inp_sentence.size)
        emb = list(map(lambda x:self.encoder(torch.LongTensor([x]).cuda()).squeeze(0), inp_sentence))
        #print('inp sen: ', inp_sentence)
        #print('emb: ', emb)
        h0 = torch.randn(self.nlayers, 1, self.h_dim, requires_grad=False).cuda()
        c0 = torch.randn(self.nlayers, 1, self.h_dim, requires_grad=False).cuda()
        output = []
        h_n = torch.zeros(len(inp_sentence), self.h_dim).cuda()
        for i in range(len(emb)):
            #print(torch.Tensor(emb[i]).unsqueeze(1).size())
            if len(emb[i]) == 0:
                opt,(h,c)=self.rnn(torch.zeros(1, 1, self.emb_dim, requires_grad=False).cuda(), (h0,c0))
            else:
                opt,(h,c)=self.rnn(emb[i].unsqueeze(1), (h0,c0))
            output.append(opt.squeeze(1))
            h_n[i]=h.squeeze(1)[self.nlayers-1]

        #print('hn:', h_n.size())

        # output is the hidden states, and h_n is the encoding result
        return output, h_n

    def init_hidden(self, bsz):
        return 0

class word_choser(nn.Module):
    def __init__(self, ntoken, ntoken_out, hidden_dim, emb_dim, node_dim, chunk_size, nlayers):
        super(word_choser, self).__init__()
        self.lockdrop = LockedDropout()
        self.inpdim = emb_dim + node_dim
        self.outdim = hidden_dim * 2
        self.dim_out = torch.nn.Parameter(torch.FloatTensor(np.zeros((self.outdim, ntoken_out))))
        ###
        #   self.attention_gcn = GCN()
        #   self.attention_pool = pool()
        ###
        self.ntoken = ntoken
        self.ntoken_out = ntoken_out
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.chunk_size = chunk_size
        self.nlayers = nlayers
        self.lstm = nn.LSTM(self.inpdim, hidden_dim, nlayers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.dim_out.data.uniform_(-initrange, initrange)

        # Training:
        #   sen_emb: emb_dim
        #   hiddens: x_len * hid_dim
        #   tree_embs: y_len * node_dim
        #
        # Evaluating:
        #   sen_emb: emb_dim
        #   hiddens: x_len * hid_dim
        #   tree_embs: node_dim
        #   ht, ct: nlayers * 1 * hid_dim
    def forward(self, sen_emb, hiddens, tree_embs, ht = False, ct = False):
        sen_len = tree_embs.size(0)

        if ht is False:
            sen_emb = sen_emb.repeat(sen_len).view(-1, self.emb_dim)
            the_inp = torch.cat((sen_emb, tree_embs),1).unsqueeze(1)
            h0 = torch.zeros(self.nlayers, 1, self.hidden_dim).cuda()
            c0 = torch.zeros(self.nlayers, 1, self.hidden_dim).cuda()
            output, (hn, cn) = self.lstm(the_inp, (h0, c0))
            output = output.cuda()
            output = output.squeeze(1)
            # output: y_len * hid_dim
            # hiddens: x_len * hid_dim
            attention = torch.mm(output, torch.t(hiddens))
            attention = F.softmax(attention, 1)
            # attention: y_len * x_len
            # hiddens: x_len * hid_dim
            attention = torch.mm(attention, hiddens)
            output = torch.cat((output, attention),1)
            output = output.mm(self.dim_out)
            output = F.softmax(output, 1)
            # output: y_len * ntoken_out
            return output
        else:
            the_inp = torch.cat((sen_emb, tree_embs)).unsqueeze(0).unsqueeze(0)
            output, (htt, ctt) = self.lstm(the_inp, (ht, ct))
            output = output.squeeze(0)
            attention = torch.mm(output, torch.t(hiddens))
            attention = F.softmax(attention, 1)
            attention = torch.mm(attention, hiddens)
            output = torch.cat((output, attention),1)
            output = output.mm(self.dim_out).squeeze(0)
            output = F.softmax(output, 0)
            return output, htt, ctt
            
            

if __name__ == "__main__":
    # (self, ntoken, node_dim, emb_dim, ntoken_out, dropout=0.1)
    pc = Pos_choser(128, 128, 128, 128)
    #se = sentence_encoder(1, 1, 1, 1, 1)
    # (self, ntoken, ntoken_out, hidden_dim, emb_dim, node_dim, chunk_size, nlayers)
    wc = word_choser(128, 128, 128, 128, 128, 16, 3)
    # def forward(self, sen_emb, hiddens, tree_embs, ht = False, ct = False):
    #   sen_emb: emb_dim
    #   hiddens: x_len * hid_dim
    #   tree_embs: y_len * node_dim
    opt = wc(torch.rand(128), torch.rand(50, 128), torch.rand(50,128))
    #   sen_emb: emb_dim
    #   hiddens: x_len * hid_dim
    #   tree_embs: node_dim
    #   ht, ct: nlayers * 1 * hid_dim
    opt, ht, ct = wc(torch.rand(128), torch.rand(50,128), torch.rand(128), torch.rand(3,1,128), torch.rand(3,1,128))