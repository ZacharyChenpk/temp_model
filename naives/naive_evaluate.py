import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import os
import operator
from nltk.translate.bleu_score import sentence_bleu
from functools import reduce

import data_pair as data
from utils import batchify, repackage_hidden
from naive_model import Pos_choser, sentence_encoder, word_choser
import tree
from tree import Tree, print_tree

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
					help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=30, metavar='N',
					help='batch size')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA')
parser.add_argument('--resume', type=str, default='',
					help='path of model to resume')

args = parser.parse_args()
args.philly = True
args.cuda = False

fn = 'corpus_fold_path'
if os.path.exists(fn) and False:
	print('Loading cached dataset...')
	corpus = torch.load(fn)
else:
	print('Producing dataset...')
	corpus = data.Corpus(args.data)
	torch.save(corpus, fn)

### Load trained model from files
def model_load(fn):
    global model_pos, model_encoder, model_word, optimizer
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'rb') as f:
        model_pos, model_encoder, model_word, optimizer = torch.load(f)

### Input a encoding of a sentence, return the decoding result and corresponding tree in timestamps
def encode2seq(model_pos, model_word, code, hiddens, corpus, strategy='greedy'):
	curtree = Tree('<start>')
	model_word.lstm.init_cellandh()
	while len(curtree.leaves(contain_single=True))>0:
		leaves, leave_inds, scores = model_pos(curtree, model_encoder, corpus.dictionary_out)
		if strategy == 'greedy':
			### Directly choose the word with highest probability
			### p_leave is the index of chosen leave
			# p = scores.index(max(scores))
			p = int(torch.argmax(scores))
			p_leave = leave_inds[p]
			### out_dist is the distribution of words probability
			out_dist = model_word(code, hiddens, p_leave)
			out_dist[corpus.dictionary_out.word2idx['<start>']] = -1
			print(out_dist)
			#print(corpus.dictionary_out)
			curtree.insert_son(p_leave, corpus.dictionary_out.idx2word[torch.argmax(out_dist)])
			curtree.make_index()
			print_tree(curtree, True)
	### Remove special token and generate sentence
	words = map(lambda x:'' if x[0]=='<' else x, curtree.horizontal_scan(contain_end=False))
	return reduce(operator.add, words), curtree

### Input a batch of sentence in words, return its generated sentence and tree
def predict_batch(model_pos, model_encoder, model_word, batch_X, corpus):
	batch_size = len(batch_X)
	print(batch_X)
	hidden_encoder = model_encoder.init_hidden(batch_size)
	hidden_outs, encodes = model_encoder(batch_X, hidden_encoder)
	

	YsYtrees = [encode2seq(model_pos, model_word, encode, hid.squeeze(1), corpus) for encode, hid in zip(encodes, hidden_outs)]
	YsYtrees = list(zip(*YsYtrees))
	YsYtrees = list(map(list, YsYtrees))

	return YsYtrees[0], YsYtrees[1]

### Calculate the total BLEU score of test data
def eval_total_bleu(model_pos, model_encoder, model_word, test_data, corpus):
	bleus = []
	for i in test_data:
		Ys, Ytrees = predict_batch(model_pos, model_encoder, model_word, [a['X'] for a in i], corpus)
		print(Ys, [a['Y'] for a in i])
		bleus.append(list(map(sentence_bleu, Ys, [a['Y'] for a in i])))

	return bleus, mean(bleus)

if __name__ == "__main__":
	### Split the data into little batches
	train_data = batchify(corpus.train, args.batch_size, args)
	val_data = batchify(corpus.valid, args.batch_size, args)
	test_data = batchify(corpus.test, args.batch_size, args)
	print(corpus.dictionary_out.idx2word)

	if args.resume:
		print('Resuming models ...')
		model_load(args.resume)

	if args.cuda:
		model_pos = model_pos.cuda()
		model_encoder = model_encoder.cuda()
		model_word = model_word.cuda()
	print('-------------------start evaluating-------------------')
	model_pos.eval()
	model_encoder.eval()
	model_word.eval()
	_, train_bleu = eval_total_bleu(model_pos, model_encoder, model_word, train_data, corpus)
	_, val_bleu = eval_total_bleu(model_pos, model_encoder, model_word, val_data, corpus)
	_, test_bleu = eval_total_bleu(model_pos, model_encoder, model_word, test_data, corpus)
	print('--------------------end evaluating--------------------')
	print('train_bleu: ', train_bleu)
	print('val_bleu: ', val_bleu)
	print('test_bleu: ', test_bleu)
