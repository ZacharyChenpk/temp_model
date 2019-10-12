from GCN import ONLSTMGraph
from ON_LSTM import ONLSTMStack

import os
import torch
import numpy as np
import torch.nn as nn
from gensim.models.word2vec import Word2Vec

def get_word_embedding(word2vec, original_sentences):
    sentences = []
    for sentence in original_sentences:
        s = []
        for word in sentence:
            print(word)
            s.append(word2vec[word])
        sentences.append(s)
    return sentences

class ModelEncoder(nn.Module):
    def __init__(self, word2vec, embedding_size, hidden_state_size, layer_size, sentence_embedding_size,
                 chunk_size, batch_size, gcn_hidden_size, gcn_output_size, dropout = 0.5):
        super(ModelEncoder, self).__init__()
        
        self.word2vec = word2vec
        self.batch_size = batch_size
        
        self.gcn_input_size = embedding_size
        self.gcn_hidden_size = gcn_hidden_size
        self.gcn_output_size = gcn_output_size
        self.dropout = dropout
        
        self.encoder = ONLSTMStack([embedding_size] + [hidden_state_size] * (layer_size - 1) + [sentence_embedding_size], chunk_size)
    
    def forward(self, sentences, hidden):
        sentences = get_word_embedding(self.word2vec, sentences)
        # sentences size: length * batch_size * word_embedding_size
        raw_output, hidden_cell, raw_outputs, outputs, distance = self.encoder(sentences, hidden)
        # hidden layer_size * batch_size * hidden_state_size
        
        distances = distance[0]
        layer_size, length, batch_size = distances.size()
        distances = distances[-1] # we use the gates of the last layer as the weights of the tree.
        distances = distances.transpose(1, 0)
        
        word_hidden_state = []
        for i in range(batch_size):
            gcn = ONLSTMGraph(distances[i], self.gcn_input_size, self.gcn_hidden_size,
                              self.gcn_output_size)
            word_hidden_state.append(gcn(sentences[i]).tolist())
        return raw_output[-1], word_hidden_state
    def init_hidden(self, batch_size):
        return self.encoder.init_hidden(batch_size)
