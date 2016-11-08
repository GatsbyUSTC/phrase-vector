import os
import gc
import random
import time
import numpy as np
import utils
from relateness import RelatenessModel
from vocab import Vocab

SEED = 93
MAX_LABEL = 5 #label y in [1,K]
NUM_EPOCHS = 10

def read_mesh(meshpath):
    meshes = []
    with open(meshpath, 'r') as f:
        for line in f:
            meshes.append(line.strip())
    return meshes

def read_dataset(ncbi_dir):
    vocab = Vocab()
    vocab.load(os.path.join(ncbi_dir, 'vocab-cased.txt'))
    
    dir_names = ['train', 'dev', 'test', 'ctd']
    sub_dirs = []
    for name in dir_names:
        sub_dirs.append(os.path.join(ncbi_dir, name))
    
    data = {}
    max_degree = 0
    for name, sub_dir in zip(dir_names, sub_dirs):
        degree, trees = utils.read_trees(os.path.join(sub_dir,'name.parents'))
        sentences = utils.read_sentences(os.path.join(sub_dir, 'name.toks'), vocab)
        meshes = read_mesh(os.path.join(sub_dir, 'mesh.txt'))
        for sentence, tree in zip(sentences, trees):
            utils.map_tokens_labels(tree, sentence, False)
        cur_dataset = {}
        cur_dataset['trees'] = trees
        cur_dataset['meshes'] = meshes
        data[name] = cur_dataset

        max_degree = max(max_degree, degree)
    
    data['max_degree'] = max_degree

    return vocab, data

def train_dataset(model, dataset, ctdset):
    
    right_num, train_num, total_loss = 0, 0, 0.0

    for lmesh, lroot in zip(dataset['meshes'], dataset['trees']):
        score_plus, score_minus = [], []
        rsents_plus, rsents_minus = [], []
        rroots_plus, rroots_minus = [], []
        # losses = []
        lsent = model.generate(lroot)
        if lmesh not in ctdset['meshes']:
            continue
            
        for rmesh, rroot in zip(ctdset['meshes'], ctdset['trees']):
            if lmesh == rmesh:
                rroots_plus.append(rroot)
                rsent = model.generate(rroot)
                rsents_plus.append(rsent)

        while len(rroots_minus) < 200:
            random_index = random.randint(0, len(ctdset['meshes'])-1)
            if ctdset['meshes'][random_index] == lmesh:
                continue
            rroot = ctdset['trees'][random_index]
            rroots_minus.append(rroot)
            rsent = model.generate(rroot)
            rsents_minus.append(rsent)

        for rsent in rsents_plus:
            score = model.getscore(lsent, rsent)
            score_plus.append(score)
        for rsent in rsents_minus:
            score = model.getscore(lsent, rsent)
            score_minus.append(score)

        if max(score_plus) > max(score_minus):
            right_num += 1
        train_num += 1

        for i in xrange(5):
            plus_index = score_plus.index(max(score_plus))
            loss = model.train(lroot, rroots_plus[plus_index], 5)
            # losses.append(loss)
            minus_index = score_minus.index(max(score_minus))
            loss = model.train(lroot, rroots_minus[minus_index], 1)
            del score_minus[minus_index]
            del rroots_minus[minus_index]
            # losses.append(loss)

        # total_loss = (total_loss * train_num + sum(losses)) / (train_num + len(losses))
        # train_num += len(losses)
        del score_plus, score_minus, rsents_plus, rsents_minus, rroots_plus, rroots_minus
        gc.collect()

    return float(right_num) / float(train_num)

def evaluate_dataset(model, dataset, ctdset):
    num_correct, num_pred = 0, 0
    rsents = []
    for rroot in ctdset['trees']:
        rsent = model.generate(rroot)
        rsents.append(rsent)
    
    for lmesh, lroot in zip(dataset['meshes'], dataset['trees']):
        if lmesh not in ctdset['meshes']:
            continue
        lsent = model.generate(lroot)
        pred_ys = []
        for rsent in rsents:
            pred_y = model.getscore(lsent, rsent)
            pred_ys.append(pred_y)

        prindex = pred_ys.index(max(pred_ys))
        if lmesh == ctdset['meshes'][prindex]:
        # if rindex == prindex:
            num_correct += 1
        num_pred += 1
        del pred_ys
        gc.collect()
    del rsents
    gc.collect()
    return float(num_correct) / float(num_pred)

def test_dataset(model, dataset, ctdset, output):
    num_correct, num_pred = 0, 0
    rsents = []
    for rroot in ctdset['trees']:
        rsent = model.generate(rroot)
        rsents.append(rsent)

    for i, lmesh, lroot in enumerate(zip(dataset['meshes'], dataset['trees'])):
        if lmesh not in ctdset['meshes']:
            continue
        lsent = model.generate(lroot)
        pred_ys = []
        for rsent in rsents:
            pred_y = model.getscore(lsent, rsent)
            pred_ys.append(pred_y)

        prindex = pred_ys.index(max(pred_ys))
        if lmesh == ctdset['meshes'][prindex]:
            num_correct += 1
        else:
            output.write(str(i) + '/t' + str(prindex) + '\t' + lmesh + '\n')
        num_pred += 1
        del pred_ys
        gc.collect()
    del rsents
    gc.collect()
    return float(num_correct) / float(num_pred)


def train():
    data_dir = '../data/ncbi'
    output_dir = '../outputs'

    curtime = time.strftime('%Y-%m-%d-%H-%M-%S')
    
    model_name = curtime + '.model'
    model_path = os.path.join(output_dir, model_name)

    output_name = curtime + '.txt'
    output_path = os.path.join(output_dir, output_name)
    output = open(output_path, 'w')

    vocab, data = read_dataset(data_dir)
    train_set, dev_set, test_set, ctd_set = data['train'], data['dev'], data['test'], data['ctd']
    max_degree = data['max_degree']
    
    output.write('train : %d\n' % len(train_set['trees']))
    output.write('dev: %d\n' % len(dev_set['trees']))
    output.write('test: %d\n\n' % len(test_set['trees']))
    
    num_emb = vocab.size()
    max_label = 5
    
    output.write('number of embeddings: %d\n' % num_emb)
    output.write('max label: %d\n' % max_label)
    output.flush()

    random.seed(SEED)
    np.random.seed(SEED)
    model = RelatenessModel(num_emb, max_degree, False)

    output.write('num_emb: %d\n' % num_emb)
    output.write('max_degree: %d\n' % max_degree)
    output.write('embedding dim: %d\n' % model.emb_dim)
    output.write('hidden dim: %d\n' % model.hidden_dim)
    output.write('learing_rate: %f\n\n' % model.learning_rate)
    output.flush()

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
        output.write('epoch: %d\n' % epoch)
        prec = train_dataset(model, train_set, ctd_set)
        output.write('avg_prec: %f\n' % prec)
        dev_score = evaluate_dataset(model, dev_set, ctd_set)
        output.write('dev score: %f\n' % dev_score)
        output.flush()
    output.write('\nstart training at ' + curtime + '\n')
    output.write('finish training at ' + time.strftime('%Y-%m-%d-%H-%M-%S') + '\n')
    test_score = evaluate_dataset(model, test_set, ctd_set)
    output.write('\ntest score: %f\n' % test_score)
    utils.save_model(model, model_path)
    output.write('\nall finished at' + time.strftime('%Y-%m-%d-%H-%M-%S'))
    output.close()

def train_test():
    data_dir = '../data/ncbi'
    output_dir = '../outputs'

    curtime = time.strftime('%Y-%m-%d-%H-%M-%S')
    
    model_name = curtime + '.model'
    model_path = os.path.join(output_dir, model_name)

    output_name = curtime + '.txt'
    output_path = os.path.join(output_dir, output_name)
    output = open(output_path, 'w')

    vocab, data = read_dataset(data_dir)
    train_set, dev_set, test_set, ctd_set = data['train'], data['dev'], data['test'], data['ctd']
    max_degree = data['max_degree']
    
    output.write('train : %d\n' % len(train_set['trees']))
    output.write('dev: %d\n' % len(dev_set['trees']))
    output.write('test: %d\n\n' % len(test_set['trees']))
    
    num_emb = vocab.size()
    max_label = 5
    
    output.write('number of embeddings: %d\n' % num_emb)
    output.write('max label: %d\n' % max_label)
    output.flush()

    random.seed(SEED)
    np.random.seed(SEED)
    model = RelatenessModel(num_emb, max_degree, False)

    output.write('trainable embeddings: %s\n' % str(model.trainable_embeddings))
    output.write('reg : %f\n' % model.reg)
    output.write('max_degree: %d\n' % max_degree)
    output.write('embedding dim: %d\n' % model.emb_dim)
    output.write('hidden dim: %d\n' % model.hidden_dim)
    output.write('learing_rate: %f\n\n' % model.learning_rate)
    output.flush()

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
        output.write('epoch: %d\n' % epoch)
        score = train_dataset(model, train_set, ctd_set)
        output.write('avg_score: %f\n' % score)
        dev_score = evaluate_dataset(model, dev_set, ctd_set)
        output.write('dev score: %f\n' % dev_score)
        output.flush()
    output.write('\nevaluate on test set\n')
    test_score = test_dataset(model, test_set, ctd_set, output)
    output.write('test score is %f\n' % test_score)
    output.write('\nevaluate on train set\n')
    train_score = test_dataset(model, train_set, ctd_set, output)
    output.write('train score is %f\n' % train_score)

    utils.save_model(model, model_path)
    output.close()   


if __name__ == '__main__':
   	train_test()