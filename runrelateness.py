import os
import random
import math
import numpy as np
import utils
from relateness import RelatenessModel
from vocab import Vocab

SEED = 93
MAX_LABEL = 5 #label y in [1,K]
NUM_EPOCHS = 10
BATCH_SIZE = 50

def read_realateness_dataset(data_dir, fine_grained=False, constituency=False):
    vocab = Vocab()
    vocab.load(os.path.join(data_dir, 'vocab-cased.txt'))
    dir_names = ['train', 'dev', 'test']
    sub_dirs = []
    for name in dir_names:
        sub_dirs.append(os.path.join(data_dir, name))

    data = {}
    max_degree = 0
    for name, sub_dir in zip(dir_names, sub_dirs):
        if constituency:
            lmax_degree, ltrees = utils.read_trees(
                os.path.join(sub_dir, 'a.cparents'),
                constituency)
            rmax_degree, rtrees = utils.read_trees(
                os.path.join(sub_dir, 'b.cpareants'),
                constituency)
        else:
            lmax_degree, ltrees = utils.read_trees(os.path.join(sub_dir, 'a.parents'))
            rmax_degree, rtrees = utils.read_trees(os.path.join(sub_dir, 'b.parents'))
        
        lsentences = utils.read_sentences(os.path.join(sub_dir, 'a.toks'), vocab)
        rsentences = utils.read_sentences(os.path.join(sub_dir, 'b.toks'), vocab)
        labels = _read_similarity(os.path.join(sub_dir, 'sim.txt'))

        cur_dataset = zip(ltrees, lsentences, rtrees, rsentences, labels)

        for ltree, lsentence, rtree, rsentence, _ in cur_dataset:
            utils.map_tokens_labels(ltree, lsentence, fine_grained)
            utils.map_tokens_labels(rtree, rsentence, fine_grained)
        

        data[name] = [(ltree, rtree, label) for ltree, _, rtree, _, label in cur_dataset]
        max_degree = max(max_degree, lmax_degree, rmax_degree)
    
    data['max_degree'] = max_degree
    return vocab, data

def _read_similarity(path):
    sims = []
    with open(path, 'r') as sim_file:
        for line in sim_file:
            sim = float(line.strip())
            sims.append(sim)
    return sims



def train():
    data_dir = '../data/ncbi'
    vocab, data = read_realateness_dataset(data_dir)
    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    max_degree = data['max_degree']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)
    
    num_emb = vocab.size()
    max_label = 5
    print 'number of embeddings', num_emb
    print 'max label', max_label

    random.seed(SEED)
    np.random.seed(SEED)
    model = RelatenessModel(num_emb, max_degree, False)

    embeddings = model.embeddings.get_value()
    glove_vecs = np.load(os.path.join(data_dir, 'glove.npy'))
    glove_words = np.load(os.path.join(data_dir, 'words.npy'))
    glove_word2idx = dict((word, i) for i, word in enumerate(glove_words))
    for i, word in enumerate(vocab.words):
        if word in glove_word2idx:
            embeddings[i] = glove_vecs[glove_word2idx[word]]
    glove_vecs, glove_words, glove_word2idx = [], [], []
    model.embeddings.set_value(embeddings)
    for epoch in xrange(NUM_EPOCHS):
        print 'epoch', epoch
        loss = train_dataset(model, train_set)
        print 'avg_loss', loss
        dev_score = evaluate_dataset(model, dev_set)
        print 'dev score', dev_score
    
    print 'finish training'
    test_score = evaluate_dataset(model, test_set)
    print 'test score', test_score
    
def train_dataset(model, dataset):
    losses = []
    for data in dataset:
        loss, pred_p = model.train(data[0], data[1], data[2])
        losses.append(loss)
    return np.mean(losses)


def evaluate_dataset(model, data):
    num_correct = 0
    for ltree, rtree, label in data:
        pred_y = model.predict(ltree, rtree)
        if abs(pred_y - label) <= 1:
            num_correct += 1
    return float(num_correct) / len(data)

if __name__ == '__main__':
    train()