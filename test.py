import numpy as np
import theano
import theano.tensor as T
import utils
from relateness import RelatenessModel
from vocab import Vocab

vocab = Vocab()
vocab.load('../data/ncbi/vocab-cased.txt')
words = np.load('../data/ncbi/words.npy')
for word in vocab.word2idx:
    if word not in words:
        print word



