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

import notree_data as data
from utils import batchify, repackage_hidden
from notree_model import Pos_choser, sentence_encoder, word_choser
import notree_tree
from notree_tree import Tree, print_tree

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../training',
					help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=30, metavar='N',
					help='batch size')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA')
parser.add_argument('--resume', type=str, default='',
					help='path of model to resume')
parser.add_argument('--threshold', type=float, default=0.3,
					help='the threshold of p to shut down the decoding and finish an output')

args = parser.parse_args()
args.philly = True
args.cuda = False

if __name__ == "__main__":
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
    global model_pos, model_encoder, model_word, optimizer, out_embedding
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'rb') as f:
        model_pos, model_encoder, model_word, optimizer, out_embedding = torch.load(f)

### Input a encoding of a sentence, return the decoding result and corresponding tree in timestamps
def encode2seq(model_pos, model_encoder, model_word, code, hiddens, corpus, threshold, strategy='greedy'):
	curtree = Tree('<start>')
	ht = torch.zeros(model_word.nlayers, 1, model_word.hidden_dim)
	ct = torch.zeros(model_word.nlayers, 1, model_word.hidden_dim)
	while True:
		if strategy == 'greedy':
			cur_graph = curtree.tree2graph(model_encoder.encoder, corpus.dictionary_out, model_pos.node_dim)
			cur_graph.the_gcn(model_pos.gcn)
			tree_emb = cur_graph.the_aggr()
			word_dist, ht, ct = model_word(code, hiddens, tree_emb, ht, ct)
			if max(word_dist) < threshold:
				break
			chosen_word = corpus.dictionary_out.idx2word[int(torch.argmax(word_dist))]
			leaves, leave_inds, pos_dist = model_pos(curtree, chosen_word, out_embedding, dictionary_out)
			tar_pos = leave_inds[int(torch.argmax(pos_dist))]
			curtree.insert_son(tar_pos, chosen_word, Training=False)
			curtree.make_index()
			print_tree(curtree, True)
	### Remove special token and generate sentence
	words = map(lambda x:'' if x[0]=='<' else x, curtree.horizontal_scan(contain_end=False))
	return reduce(operator.add, words), curtree

### Input a batch of sentence in words, return its generated sentence and tree
def predict_batch(model_pos, model_encoder, model_word, batch_X, corpus, threshold):
	batch_size = len(batch_X)
	print(batch_X)
	# hidden_encoder = model_encoder.init_hidden(batch_size)
	# waiting
	#	sen_embs: bsz * emb_dim
	#	hiddens: bsz * x_len * hid_dim
	hiddens, sen_embs = model_encoder(batch_X)
	# waiting

	YsYtrees = [encode2seq(model_pos, model_encoder, model_word, encode, hid.squeeze(1), corpus, threshold) for encode, hid in zip(sen_embs, hiddens)]
	YsYtrees = list(zip(*YsYtrees))
	YsYtrees = list(map(list, YsYtrees))

	return YsYtrees[0], YsYtrees[1]

### Calculate the total BLEU score of test data
def eval_total_bleu(model_pos, model_encoder, model_word, test_data, test_ans, corpus):
	bleus = []
	assert(len(test_data)==len(test_ans))
	for i in range(len(test_data)):
		Ys, Ytrees = predict_batch(model_pos, model_encoder, model_word, test_data[i], corpus, args.threshold)
		print(Ys, test_ans[i])
		bleus.append(list(map(sentence_bleu, Ys, test_ans[i])))

	return bleus, mean(bleus)

if __name__ == "__main__":
	### Split the data into little batches
	#train_data_X = batchify(corpus.train[0], args.batch_size, args)
	#val_data_X = batchify(corpus.valid[0], eval_batch_size, args)
	test_data_X = batchify(corpus.test[0], args.batch_size, args)
	#train_data_Y = batchify(corpus.train[1], args.batch_size, args)
	#val_data_Y = batchify(corpus.valid[1], eval_batch_size, args)
	test_data_Y = batchify(corpus.test[1], args.batch_size, args)
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
	#_, train_bleu = eval_total_bleu(model_pos, model_encoder, model_word, train_data, corpus)
	#_, val_bleu = eval_total_bleu(model_pos, model_encoder, model_word, val_data_X, val_data_Y, corpus)
	_, test_bleu = eval_total_bleu(model_pos, model_encoder, model_word, test_data_X, test_data_Y, corpus)
	print('--------------------end evaluating--------------------')
	#print('train_bleu: ', train_bleu)
	#print('val_bleu: ', val_bleu)
	print('test_bleu: ', test_bleu)
