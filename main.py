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
from random import sample, random, randint
from functools import reduce

import data_pair as data
from utils import batchify, repackage_hidden
from model import Pos_choser, sentence_encoder, word_choser
from evaluate import predict_batch
import tree
from tree import random_seq, print_tree

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--hidsize', type=int, default=512,
					help='size of hidden states in lstm')
parser.add_argument('--nodesize', type=int, default=512,
					help='size of nodes presentation in tree/graph')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=16,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--poslr', type=float, default=30,
                    help='initial pos learning rate')
parser.add_argument('--encoderlr', type=float, default=30,
                    help='initial encoder learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

args = parser.parse_args()
args.tied = True

def model_save(fn):
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'wb') as f:
        torch.save([model_pos, model_encoder, model_word, optimizer], f)


def model_load(fn):
    global model_pos, model_encoder, model_word, optimizer
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'rb') as f:
        model_pos, model_encoder, model_word, optimizer = torch.load(f)

import hashlib

always_producing = True

fn = 'corpus_fold_path'
if os.path.exists(fn) and not always_producing:
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
print(corpus.train)

train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

ntokens = len(corpus.dictionary)
ntokens_out = len(corpus.dictionary_out)

model_pos = Pos_choser(ntokens, args.nodesize, args.dropout)
model_encoder = sentence_encoder(ntokens, args.hidsize, args.emsize, args.nlayers, args.chunk_size, wdrop=0, dropouth=args.dropout)
model_word = word_choser(ntokens, ntokens_out, args.hidsize, args.emsize, args.chunk_size, args.nlayers)

if args.resume:
	print('Resuming models ...')
	model_load(args.resume)

if args.cuda:
	model_pos = model_pos.cuda()
	model_encoder = model_encoder.cuda()
	model_word = model_word.cuda()

params = list(model_encoder.parameters()) + list(model_word.parameters())
pos_params =  list(model_pos.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params + pos_params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

#############################################
# Training
#############################################

def single_tree_loss(tree, true_ans):
	leaves, leave_inds, scores = model_pos(tree, model_encoder, corpus.dictionary_out)
	ans_dis = torch.zeros(len(leaves))
	ans_dis[leave_inds.find(true_ans)] = 1
	return F.kl_div(scores, ans_dis)

def single_sen_decode_loss(encode, hiddens, true_pos, word_ans):
	model_word.lstm.init_cellandh()
	out_dist = model_word(encode, hiddens, true_pos)
	return F.kl_div(out_dist, word_ans)


def batch_loss(X, Y, Y_tree):
	decoder_loss = 0.0
	pos_loss = 0.0
	ntokens = len(corpus.dictionary)
	ntokens_out = len(corpus.dictionary_out)
	hidden_encoder = model_encoder.init_hidden(args.batch_size)
	hidden_outs, layer_outs, _, opts = model_encoder(X, hidden_encoder)
	encodes = layer_outs[-1][1]

	batch_tree_ret = list(map(random_seq, Y_tree))
	batch_tree_ret = list(zip(*batch_tree_ret))
	batch_tree_ret = list(map(list, batch_tree_ret))
	print('batch_tree_ret:')
	for aa in batch_tree_ret:
		print(aa)
	raw_lenseqs = list(map(len, batch_tree_ret[0]))
	lenseqs = [0]*(len(raw_lenseqs)-1)
	for i in range(len(raw_lenseqs)-1):
		lenseqs[i+1]=lenseqs[i]+raw_lenseqs[i]
	seqs = reduce(operator.add, batch_tree_ret[0])
	indseqs = list(reduce(operator.add, batch_tree_ret[1]))
	wordseqs = reduce(operator.add, batch_tree_ret[2])
	treeseqs = reduce(operator.add, batch_tree_ret[3])
	pos_loss = sum(map(single_tree_loss, treeseqs, indseqs))

	wordseqs = map(corpus.dictionary_out.word2idx.__getitem__, wordseqs)
	word_onehot = torch.zeros(args.batch_size, ntokens_out).scatter_(1, wordseqs, 1)
	for i in range(len(lenseqs)-1):
		decoder_loss += single_sen_decode_loss(encodes[i], hidden_outs[lenseqs[i]:lenseqs[i+1]], indseqs[i], word_onehot[i])

	return pos_loss, decoder_loss

optimizer = None
args.wdecay = 0
# Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
if args.optimizer == 'sgd':
    optimizer_pos = torch.optim.SGD(pos_params, lr=args.poslr, weight_decay=args.wdecay)
    optimizer_encoder = torch.optim.SGD(params, lr=args.encoderlr, weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer_pos = torch.optim.Adam(pos_params, lr=args.poslr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
    optimizer_encoder = torch.optim.Adam(params, lr=args.encoderlr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
    scheduler_pos = lr_scheduler.ReduceLROnPlateau(optimizer_pos, 'min', 0.5, patience=2, threshold=0)
    scheduler_encoder = lr_scheduler.ReduceLROnPlateau(optimizer_encoder, 'min', 0.5, patience=2, threshold=0)

def train_one_epoch(epoch):
	# We can use dropout in training?
	total_loss = 0.
	start_time = time.time()
	ntokens = len(corpus.dictionary)
	ntokens_out = len(corpus.dictionary_out)
	hidden_encoder = model_encoder.init_hidden(args.batch_size)
	hidden_pos = model_pos.init_hidden()

	for i in train_data:
		X = torch.LongTensor([x['X'] for x in i])
		Y = torch.LongTensor([x['Y'] for x in i])
		Y_tree = [x['Y_tree'] for x in i]
		if args.cuda:
			X = X.cuda()
			Y = Y.cuda()

		model_pos.train()
		model_encoder.train()
		model_word.train()
		# hidden_encoder = repackage_hidden(hidden_encoder)
		# hidden_pos = repackage_hidden(hidden_pos)
		optimizer_pos.zero_grad()
		optimizer_encoder.zero_grad()

		pos_loss, decoder_loss = batch_loss(X, Y, Y_tree)
		pos_loss.backward()
		decoder_loss.backward()

		if args.clip: 
			torch.nn.utils.clip_grad_norm_(params, args.clip)

		optimizer_pos.step()
		optimizer_encoder.step()

		if random()>0.7:
			model_pos.eval()
			model_encoder.eval()
			model_word.eval()
			the_sample = randint(0, len(X)-1)
			Ys, Ytrees = predict_batch(model_pos, model_encoder, model_word, X[the_sample], corpus)
			print('Trigger a show!')
			print('input sentence:', X[the_sample])
			print('true answer:', Y[the_sample])
			print('output sentence:', Ys[0])
			print_tree(Ytrees[0], show_index=True)

	print('epoch: {0}, pos_loss:{1}, decoder_loss:{2}, sentence/s: {3}'.format(epoch, pos_loss, decoder_loss, int(len(X)/(time.time()-start_time))))
	global_pos_losses.append(pos_loss)
	global_decoder_losses.append(decoder_loss)

	if epoch % args.save_every == 0:
		print('saving checkpoint at epoch {0}'.format(epoch))
		model_save('models.checkpoint')


#lr = args.lr
global_pos_losses = []
global_decoder_losses = []
stored_loss = 100000000

# use Ctrl+C to break out of training at any point
try:
	print('traindata', train_data)
	for epoch in range(1, args.epochs + 1):
		train_one_epoch(epoch)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    model_save('models')
    print('| End of training | pos loss/epoch {:5.2f} | decoder ppl/epoch {:5.2f}'.format(mean(global_pos_losses), mean(global_decoder_losses)))
