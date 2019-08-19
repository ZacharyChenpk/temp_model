import os
import torch
import sys

from collections import Counter

import nltk
import torch
### Add the data path to syspath
sys.path.append('./data/nltk_data')
from nltk.corpus import treebank

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    """
        read in dataset and tokenize the corpus
    """
    def __init__(self, path):
    	### Input language dictionary
        self.dictionary = Dictionary()
        ### Output language dictionary
        self.dictionary_out = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train_x.txt', 'train_y.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid_x.txt', 'valid_y.txt'))
        self.test = self.tokenize(os.path.join(path, 'test_x.txt', 'test_y.txt'))

    def pairtoken(self, the_tuple):
        x, y = the_tuple
        xwords = x.split() + ['<eos>']
        ywords = y.split() + ['<eos>']
        return (map(self.dictionary.word2idx.__getitem__, xwords), map(self.dictionary.word2idx.__getitem__, ywords))

        ### Tokenizes two text files and add to the dictionary
        ### We stored source sentences in PATH and target sentences in TARGET_PATH
    def tokenize(self, path, target_path):
        assert os.path.exists(path)
        assert os.path.exists(target_path)
        ### Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        with open(target_path, 'r') as f:
            tar_tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tar_tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        ### Tokenize file content
        zipped = list(zip(open(path, 'r'), open(target_path, 'r')))

        return map(self.pairtoken, zipped)
