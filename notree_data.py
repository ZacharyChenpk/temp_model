import os
import torch
import sys
import numpy as np

from collections import Counter

import nltk
import torch
### Add the data path to syspath
sys.path.append('../training')
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
        self.train = self.tokenize(os.path.join(path, 'train_x.txt'), os.path.join(path, 'train_y.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid_x.txt'), os.path.join(path, 'valid_y.txt'))
        self.test = self.tokenize(os.path.join(path, 'test_x.txt'), os.path.join(path, 'test_y.txt'))
        print(type(self.train))
    '''
    def pairtoken(self, the_tuple, the_tree):
        x, y = the_tuple
        xwords = x.split()
        ywords = y.split()
        return {'X': list(map(self.dictionary.word2idx.__getitem__, xwords)), 'Y': list(map(self.dictionary_out.word2idx.__getitem__, ywords)), 'Y_tree': the_tree}
    '''
    def sen_tokenize(self, sen, is_out = False):
        words = sen.split()
        if not is_out:
            return [self.dictionary.word2idx[i] for i in words]
        else:
            return [self.dictionary_out.word2idx[i] for i in words]

        ### Tokenizes two text files and add to the dictionary
        ### We stored source sentences in PATH and target sentences in TARGET_PATH
    def tokenize(self, path, target_path):
        assert os.path.exists(path)
        assert os.path.exists(target_path)
        ### Add words to the dictionary
        self.dictionary.add_word('<start>')
        self.dictionary.add_word('<eod>')
        self.dictionary_out.add_word('<start>')
        self.dictionary_out.add_word('<eod>')
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        with open(target_path, 'r') as f:
            tar_tokens = 0
            for line in f:
                words = line.split()
                tar_tokens += len(words)
                for word in words:
                    self.dictionary_out.add_word(word)

        ### Tokenize file content
        '''
        zipped = list(zip(open(path, 'r'), open(target_path, 'r')))
        #print(zipped)

        return np.asarray(list(map(self.pairtoken, zipped, torch.load(tree_path))), dtype = tuple)
        '''

        ret1, ret2 = list(map(self.sen_tokenize, open(path, 'r'))), list(map(lambda x:self.sen_tokenize(x, True), open(target_path, 'r')))
        assert(len(ret1)==len(ret2))

        return np.asarray(ret1), np.asarray(ret2), list(zip(open(path, 'r')))
