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

import notree_data as data
from utils import batchify, repackage_hidden
from notree_model import Pos_choser, sentence_encoder, word_choser
from notree_evaluate import predict_batch
import notree_tree as tree
from notree_tree import behave_seq_gen, print_tree
from encoder import ModelEncoder
from gensim.models.word2vec import Word2Vec

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=512,
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
parser.add_argument('--poslr', type=float, default=0.003,
                    help='initial pos learning rate')
parser.add_argument('--encoderlr', type=float, default=0.003,
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
args.cuda = False
args.philly = True
args.save_every = 1000

def model_save(fn):
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'wb') as f:
        torch.save([model_pos, model_encoder, model_word, optimizer, out_embedding], f)


def model_load(fn):
    global model_pos, model_encoder, model_word, optimizer, out_embedding
    if args.philly:
        fn = os.path.join(os.getcwd(), fn)
    with open(fn, 'rb') as f:
        model_pos, model_encoder, model_word, optimizer, out_embedding = torch.load(f)

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
# print(corpus.train)

train_data_X = batchify(corpus.train[0], args.batch_size, args)
val_data_X = batchify(corpus.valid[0], eval_batch_size, args)
test_data_X = batchify(corpus.test[0], test_batch_size, args)
train_data_Y = batchify(corpus.train[1], args.batch_size, args)
val_data_Y = batchify(corpus.valid[1], eval_batch_size, args)
test_data_Y = batchify(corpus.test[1], test_batch_size, args)

ntokens = len(corpus.dictionary)
ntokens_out = len(corpus.dictionary_out)

word2vec = Word2Vec(size = args.emsize)
word2vec.build_vocab(corpus.train[2], min_count = 1)
word2vec.train(corpus.train[2], total_examples = word2vec.corpus_count, epochs = word2vec.iter)

model_pos = Pos_choser(ntokens, args.nodesize, args.emsize, len(corpus.dictionary_out.idx2word))
model_encoder = ModelEncoder(word2vec, args.emsize, args.hidsize, args.nlayers, args.emsize,
                 args.chunk_size, args.batch_size, args.hidsize, args.emsize, dropout = args.dropout)
model_word = word_choser(ntokens, ntokens_out, args.hidsize, args.emsize, args.nodesize, args.chunk_size, args.nlayers)
out_embedding = nn.Embedding(ntokens_out, args.emsize)

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
# print('Args:', args)
# print('Model total parameters:', total_params)

#############################################
# Training
#############################################

def batch_loss(sentences, X, Y):
    assert(len(X)==args.batch_size)
    # waiting
    #   sen_embs: bsz * emb_dim
    #   hiddens: bsz * x_len * hid_dim
    init_hidden = model_encoder(args.batch_size)
    sen_embs, hiddens = model_encoder(sentences, init_hidden)
    print("size of sen_embs", sen_embs.size())
    print("size of hiddens", len(hiddens), len(hiddens[0]))
    # waiting
    word_loss = 0.0
    pos_loss = 0.0
    for i in range(len(X)):
        sen_emb = sen_embs[i]
        hidden = hiddens[i]
        x_len = len(X[i])
        y_len = len(Y[i])
        tar_sen = [corpus.dictionary_out.idx2word[a] for a in Y[i]]
        ans_ind, choose_words, trees_before_insert = behave_seq_gen(tar_sen)
        graphs_before_insert = [a.tree2graph(out_embedding, corpus.dictionary_out, args.nodesize) for a in trees_before_insert]
        # gcns: y_len * node_num * node_dim
        # aggrs: y_len * node_dim
        gcns = [a.the_gcn(model_pos.gcn) for a in graphs_before_insert]
        aggrs = torch.FloatTensor([a.the_aggr().tolist() for a in graphs_before_insert])
        #print(sen_embs.size())
        #print(sen_emb.size())
        output = model_word(sen_emb, hidden, aggrs)
        able_words = [a.able_words() for a in trees_before_insert]
        able_inds = [[corpus.dictionary_out.word2idx[b] for b in a] for a in able_words]
        prob_vals = [1.0/len(a) for a in able_inds]
        ans_dist = torch.zeros(y_len, ntokens_out, requires_grad=False)
        for a in range(y_len):
            ans_dist[a][able_inds[a]] = prob_vals[a]
        word_loss = word_loss + F.binary_cross_entropy(output, ans_dist)

        for a in range(y_len):
            able_pos = list(map(lambda w: trees_before_insert[a].find_able_pos(w), able_words[a]))
            #print(able_pos)
            def calculate_loss(x):
                _, leave_inds, scores = model_pos(trees_before_insert[a], able_words[a][x], out_embedding, corpus.dictionary_out)
                prob_vals = 1.0/len(able_pos[x])
                ans_dist_pos = torch.zeros(scores.size(), requires_grad=False)
                ids = [leave_inds.index(the_id) for the_id in able_pos[x]]
                ans_dist_pos[ids] = prob_vals
                #print(scores, "###", ans_dist_pos)
                #print_tree(trees_before_insert[a])
                return F.binary_cross_entropy(scores, ans_dist_pos)

            pos_loss = pos_loss + sum(map(calculate_loss, range(len(able_words[a])))) / len(able_words[a])
    return word_loss, pos_loss


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

    for i in range(len(train_data_X)):
        X = train_data_X[i]
        Y = train_data_Y[i]
        print("size of X", len(X), len(X[0]))
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

        pos_loss, word_loss = batch_loss(sentences, X, Y)
        pos_loss.backward()
        word_loss.backward()

        if args.clip: 
            torch.nn.utils.clip_grad_norm_(params, args.clip)

        optimizer_pos.step()
        optimizer_encoder.step()
        '''
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
        '''
        print('epoch: {0}, pos_loss:{1}, word_loss:{2}, sentence/s: {3}'.format(epoch, pos_loss, word_loss, int(len(X)/(time.time()-start_time))))
        global_pos_losses.append(pos_loss)
        global_decoder_losses.append(word_loss)

    if epoch % args.save_every == 0:
        print('saving checkpoint at epoch {0}'.format(epoch))
        model_save('models.checkpoint')


#lr = args.lr
global_pos_losses = []
global_decoder_losses = []
stored_loss = 100000000

# use Ctrl+C to break out of training at any point
try:
    print('traindataX', train_data_X)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(epoch)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    model_save('models')
    print('| End of training | pos loss/epoch {:5.2f} | decoder ppl/epoch {:5.2f}'.format(torch.mean(torch.Tensor(global_pos_losses)), torch.mean(torch.Tensor(global_decoder_losses))))

