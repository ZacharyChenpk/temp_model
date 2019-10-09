'''
By tsy
'''
from GCN import ONLSTMGraph
from ON_LSTM import ONLSTMStack

import os
import torch
import numpy as np
import torch.nn as nn

class ModelEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_state_size, layer_size, sentence_embedding_size,
                 chunk_size, gcn_input_size, gcn_hidden_size, gcn_output_size, dropout = 0.5):
        super(ModelEncoder, self).__init__()
        
        self.gcn_input_size = gcn_input_size
        self.gcn_hidden_size = gcn_hidden_size
        self.gcn_output_size = gcn_output_size
        self.dropout = dropout
        
        self.encoder = ONLSTMStack([embedding_size] + [hidden_state_size] * (layer_size - 1) + [sentence_embedding_size], chunk_size)
    
    def forward(self, sentences, hidden):
        raw_output, hidden_cell, raw_outputs, outputs, distance = self.encoder(sentences, hidden)
        # hidden layer_size * batch_size * hidden_state_size
        
        distances = distance[0]
        layer_size, length, batch_size = distances.size()
        distances = distances[-1] # we use the gates of the last layer as the weights of the tree.
        distances = distances.transpose(1, 0)
        
        word_hidden_state = []
        for i in range(len(batch_size)):
            gcn = ONSLTMGraph(distances[i], self.gcn_input_size, self.gcn_hidden_size,
                              self.gcn_output_size)
            word_hidden_state.append(gcn(torch.zeros([length, self.gcn_input_size])))
            # word_hidden_state.append(gcn(words_embedding))
        return raw_output[-1], word_hidden_state
    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)
