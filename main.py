import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os

import data_pair
from utils import batchify, repackage_hidden

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