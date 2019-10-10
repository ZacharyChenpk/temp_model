from gensim.models.word2vec import Word2Vec

import os
import sys
import torch

from encoder import ModelEncoder
from utils import batchify
import data_pair as data

def get_sentence(input_file):
    print("read data")
    assert os.path.exists(input_file)
    sentences = []
    with open(input_file, "r") as f:
        tot = 0
        for line in f:
            s = line.split() + ['<eos>']
            sentences.append(s)
            tot = tot + 1
            if tot > 10:
                break
    return sentences

def get_word_embedding(word2vec, original_sentences):
    sentences = []
    for sentence in original_sentences:
        s = []
        for word in sentence:
            s.append(word2vec[word])
        sentences.append(s)
    return sentences

sentences = get_sentence("../training/europarl-v7.cs-en.cs")

word2vec = Word2Vec(size = 10)
word2vec.build_vocab(sentences, min_count = 1)
word2vec.train(sentences, total_examples = word2vec.corpus_count, epochs = word2vec.iter)

sentences = get_word_embedding(word2vec, sentences)

word = []
for sentence in sentences:
    word = word + sentence
input = torch.Tensor(word)
input = input.reshape(-1, 2, 10)
print(input[0])

encoder = ModelEncoder(10, 10, 5, 10, 5, 10, 8);
print("train")
hidden = encoder.init_hidden(2)
# print(hidden)
# sys.exit(0)
# print("train")
print(encoder(input, hidden))
