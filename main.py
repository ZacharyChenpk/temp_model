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