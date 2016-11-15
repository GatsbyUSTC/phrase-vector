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

def read_name(namepath):
    names = []
    with open(namepath, 'r') as f:
        for line in f:
            names.append(line.strip())
    return names

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
        names = read_name(os.path.join(sub_dir, 'name.txt'))
        for sentence, tree in zip(sentences, trees):
            utils.map_tokens_labels(tree, sentence, False)
        cur_dataset = {}
        cur_dataset['trees'] = trees
        cur_dataset['meshes'] = meshes
        cur_dataset['names'] = names
        cur_dataset['wso'] = []
        cur_dataset['wsw'] = []
        data[name] = cur_dataset

        max_degree = max(max_degree, degree)
    
    data['max_degree'] = max_degree

    return vocab, data

def load_wrongsamples(dataset, wspath):
    with open(wspath, 'r') as wsf:
        for line in wsf:
            oid, wid = line.strip().split('\t')
            dataset['wso'].append(int(oid))
            dataset['wsw'].append(int(wid))
    
def save_wrongsamples(dataset, wspath):
    with open(wspath, 'w') as wsf:
        for oid, wid in zip(dataset['wso'], dataset['wsw']):
            wsf.write(str(oid) + '\t' + str(wid) + '\n')


def train_embeddings(model, ctdset):
    for i, (lmesh, lroot) in enumerate(zip(ctdset['meshes'], ctdset['trees'])):
        for j in xrange(i+1, len(ctdset['meshes'])):
            rmesh, rroot = ctdset['meshes'][j], ctdset['trees'][j]
            if lmesh != rmesh:
                break
            model.train_em(lroot, rroot, 5)
        
        for r in xrange(5):
            ri = random.randint(0, len(ctdset['meshes']) - 1)
            rmesh, rroot = ctdset['meshes'][ri], ctdset['trees'][ri]
            if rmesh == lmesh:
                continue
            model.train_em(lroot, rroot, 1)

def evaluate_dataset(model, dataset, ctdset, output):
    num_correct, num_pred = 0, 0
    rsents = []
    for rroot in ctdset['trees']:
        rsent = model.generate(rroot)
        rsents.append(rsent)
    
    cm_num_correct, cm_num_total = 0, 0
    no_that_mesh = 0
    del dataset['wso']
    del dataset['wsw']
    dataset['wso'], dataset['wsw'] = [], []
    for i, (lname, lmesh, lroot) in enumerate(zip(dataset['names'], dataset['meshes'], dataset['trees'])):
        if lmesh not in ctdset['meshes']:
            no_that_mesh += 1            
            continue
        if lname in ctdset['names']:
            if ctdset['meshes'][ctdset['names'].index(lname)] == lmesh:
                cm_num_correct += 1
            cm_num_total += 1
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
           dataset['wso'].append(i)
           dataset['wsw'].append(prindex)
        num_pred += 1
        del pred_ys
        gc.collect()
    output.write("no that mesh: %d\n" % no_that_mesh)
    output.write('cm_num_correct: %d\n' % cm_num_correct)
    output.write('cm_num_total: %d\n' % cm_num_total)
    output.write('num_pred_correct: %d\n' % num_correct)
    output.write('num_pred_total: %d\n' % num_pred)
    del rsents
    gc.collect()
    return float(num_correct) / float(num_pred)

def train_test():
    data_dir = '../data/ncbi'
    output_dir = '../outputs'

    curtime = time.strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(output_dir, curtime)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    model_name = curtime + '.model'
    model_path = os.path.join(output_dir, model_name)

    output_name = curtime + '.ouput'
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
    
    old_model_path =  '../outputs/2016-11-12-13-21-57/2016-11-12-13-21-57.model'
    utils.load_model(model, old_model_path)

    output.write('trainable embeddings: %s\n' % str(model.trainable_embeddings))
    output.write('reg : %f\n' % model.reg)
    output.write('max_degree: %d\n' % max_degree)
    output.write('embedding dim: %d\n' % model.emb_dim)
    output.write('hidden dim: %d\n' % model.hidden_dim)
    output.write('learing_rate: %f\n\n' % model.learning_rate)
    output.flush()
    
    for epoch in xrange(NUM_EPOCHS):
        output.write('epoch: %d\n' % epoch)
        score = train_embeddings(model, ctd_set)
        score = evaluate_dataset(model, dev_set, ctd_set, output)
        output.write('dev score: %f\n' % score)
        output.flush()
    
    utils.save_model(model, model_path)
    output.write('\nevaluate on train set')
    score = evaluate_dataset(model, train_set, ctd_set, output)
    output.write('train score: %f\n' % score)
    output.write('\nevaluate on test set\n')
    test_score = evaluate_dataset(model, test_set, ctd_set, output)
    output.write('\ntest score is %f\n' % test_score)
    
    for name in ['train', 'dev', 'test']:
        ws_name = curtime + '.ws' + name
        ws_path = os.path.join(output_dir, ws_name)
        save_wrongsamples(data[name], ws_path)

    output.close()

if __name__ == '__main__':
   	train_test()