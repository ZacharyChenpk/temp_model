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
<<<<<<< HEAD
    #print(data)
    #print(type(data))
=======
    # print(data)
    # print(type(data))
>>>>>>> f3379fb0be3fa3ad2827f614fec0cab98b62f461
    # Evenly divide the data across the bsz batches.
    data = data.reshape((bsz, -1)).T ###.contiguous()
    '''
    data = torch.Tensor(data)
    if args.cuda:
        data = data.cuda()
    '''
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
