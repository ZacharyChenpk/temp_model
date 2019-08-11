import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os

import data_pair as data
from utils import batchify, repackage_hidden
from model import Pos_choser, sentence_encoder, word_choser

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--hidsize', type=int, defalut=512,
					help='size of hidden states in lstm')
parser.add_argument('--nodesize', type=int, defalut=512,
					help='size of nodes presentation in tree/graph')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
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
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import hashlib

fn = 'corpus_fold_path'
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

ntokens = len(corpus.dictionary)

model_pos = Pos_choser(ntokens, args.nodedim, args.dropout)
model_encoder = sentence_encoder(ntokens, args.hidsize, args.emsize, args.nlayers, args.chunk_size, wdrop=0, dropouth=args.dropout)
model_word = word_choser(ntokens, args.hidsize, args.emsize, args.chunk_size, args.nlayers)

if args.resume
	print('Resuming models ...')

if args.cuda:
	model_pos = model_pos.cuda()
	model_encoder = model_encoder.cuda()
	model_word = model_word.cuda()

params = list(model_pos.parameters()) + list(model_encoder.parameters()) + list(model_word.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

#############################################
# Training
#############################################

def batch_loss(X, Y, Y_tree):
	decoder_loss = 0.0
	pos_loss = 0.0
	ntokens = len(corpus.dictionary)
	hidden_encoder = model_encoder.init_hidden(args.batch_size)
	hidden_outs, layer_outs = model_encoder(X, hidden_encoder)
	encodes = layer_outs[-1][1]
	batch_tree_ret = map(random_seq, Y_tree)
	batch_tree_ret = list(zip(*batch_tree_ret))
	batch_tree_ret = map(list, batch_tree_ret)
	seqs = batch_tree_ret[0]
	indseqs = batch_tree_ret[1]
	wordseqs = batch_tree_ret[2]
	treeseqs = batch_tree_ret[3]
