import torch
import numpy as np

### Wraps hidden states in new Tensors to detach them from their history
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

### Transfer the data into batches
def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:(nbatch * bsz)]
    len_list = [len(s) for s in data]
    max_len = max(len_list)
    
    #print(data)
    #print(type(data))
    # Evenly divide the data across the bsz batches.
    data2 = [d+[1]*(max_len-len(d)) for d in data]
    data2 = torch.LongTensor(data2).cuda()
    data2 = data2.reshape((nbatch, bsz, max_len)) ###.contiguous()
    '''
    data = torch.Tensor(data)
    if args.cuda:
        data = data.cuda()
    '''
    return data2, len_list


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
