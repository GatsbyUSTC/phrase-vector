import numpy as np
import theano
import theano.tensor as T
import utils
from relateness import RelatenessModel
from vocab import Vocab
import os

ncbi_dir = '../data/ncbi'
names = ['train', 'dev', 'test', 'ctd']
vocab = {}
for name in names:
    vocab[name] = set()
    path = os.path.join(ncbi_dir, name)
    path = os.path.join(path, 'name.toks')
    with open(path, 'r') as f:
        for line in f:
            vocab[name] |= set(line.lower().split())
in_count, outcount, total = 0, 0, 0
for word in vocab['dev']:
    if word in vocab['ctd']:
        in_count += 1
    else:
        outcount += 1
    total += 1
print in_count, outcount, total


