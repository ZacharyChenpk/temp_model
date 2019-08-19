import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import os
from nltk.translate.bleu_score import sentence_bleu

import data_pair as data
from utils import batchify, repackage_hidden
from model import Pos_choser, sentence_encoder, word_choser
import tree
from main import model_save, model_load

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

fn = 'corpus_fold_path'
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

### Split the data into little batches
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, args.batch_size, args)
test_data = batchify(corpus.test, args.batch_size, args)

### Load trained model from files
if args.resume:
	print('Resuming models ...')
	model_load(args.resume)

if args.cuda:
	model_pos = model_pos.cuda()
	model_encoder = model_encoder.cuda()
	model_word = model_word.cuda()

### Input a encoding of a sentence, return the decoding result and corresponding tree in timestamps
def encode2seq(model_pos, model_word, code, hiddens, corpus, strategy='greedy'):
	curtree = Tree('<start>')
	while len(curtree.leaves(contain_single=True))>0:
		leaves, leave_inds, scores = model_pos(curtree)
		if strategy == 'greedy':
			### Directly choose the word with highest probability
			### p_leave is the index of chosen leave
			p = scores.index(max(scores))
			p_leave = leave_inds[p]
			### out_dist is the distribution of words probability
			out_dist = model_word(encode, hiddens, p_leave)
			curtree.insert_son(p_leave, corpus.dictionary_out.idx2word[out_dist.index(max(out_dist))])
			curtree.make_index()
	### Remove special token and generate sentence
	words = map(lambda x:'' if x[0]=='<' else x, curtree.horizontal_scan(contain_end=False))
	return reduce(operator.add, words), curtree

### Input a batch of sentence in words, return its generated sentence and tree
def predict_batch(model_pos, model_encoder, model_word, batch_X, corpus):
	batch_size = len(batch_X)
	hidden_encoder = model_encoder.init_hidden(batch_size)
	hidden_outs, layer_outs = model_encoder(batch_X, hidden_encoder)
	encodes = layer_outs[-1][1]

	Ys, Ytrees = map(encode2seq, encodes)
	return Ys, Ytrees

### Calculate the total BLEU score of test data
def eval_total_bleu(model_pos, model_encoder, model_word, test_data, corpus):
	bleus = []
	for i in test_data:
		Ys, Ytrees = predict_batch(model_pos, model_encoder, model_word, i[:]['X'], corpus)
		bleus.append(map(sentence_bleu, Ys, i[:]['Y']))

	return bleus, mean(bleus)

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
