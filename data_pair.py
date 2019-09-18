import os
import torch
import sys
import numpy as np

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
		self.train = self.tokenize(os.path.join(path, 'train_x.txt'), os.path.join(path, 'train_y.txt'), os.path.join(path, 'train_tree.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid_x.txt'), os.path.join(path, 'valid_y.txt'), os.path.join(path, 'valid_tree.txt'))
		self.test = self.tokenize(os.path.join(path, 'test_x.txt'), os.path.join(path, 'test_y.txt'), os.path.join(path, 'test_tree.txt'))
		print(type(self.train))

	def pairtoken(self, the_tuple, the_tree):
		x, y = the_tuple
		xwords = x.split()
		ywords = y.split()
		return {'X': list(map(self.dictionary.word2idx.__getitem__, xwords)), 'Y': list(map(self.dictionary_out.word2idx.__getitem__, ywords)), 'Y_tree': the_tree}

		### Tokenizes two text files and add to the dictionary
		### We stored source sentences in PATH and target sentences in TARGET_PATH
	def tokenize(self, path, target_path, tree_path):
		assert os.path.exists(path)
		assert os.path.exists(target_path)
		assert os.path.exists(tree_path)
		### Add words to the dictionary
		self.dictionary.add_word('<start>')
		self.dictionary.add_word('<end>')
		self.dictionary_out.add_word('<start>')
		self.dictionary_out.add_word('<end>')
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
					self.dictionary_out.add_word(word)

		### Tokenize file content
		zipped = list(zip(open(path, 'r'), open(target_path, 'r')))
		#print(zipped)

		return np.asarray(list(map(self.pairtoken, zipped, torch.load(tree_path))), dtype = tuple)
