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
        sentences = utils.read_sentences(os.path.join(sub_dir, 'name.tokss'), vocab)
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

def train_dataset(model, dataset, ctdset):
    
    right_num, train_num = 0, 0

    for i, (lmesh, lroot) in enumerate(zip(dataset['meshes'], dataset['trees'])):
        if lmesh not in ctdset['meshes']:
            continue
        score_plus, score_minus = [], []
        rsents_plus, rsents_minus = [], []
        rroots_plus, rroots_minus = [], []
        
        lsent = model.generate(lroot)
        for rmesh, rroot in zip(ctdset['meshes'], ctdset['trees']):
            if lmesh != rmesh:
                continue
            rroots_plus.append(rroot)
            rsent = model.generate(rroot)
            rsents_plus.append(rsent)
        
        if i in dataset['wso']:
            index = dataset['wso'].index(i)
            wid = dataset['wsw'][index]
            first_index, last_index = max(0, wid - 20), min(wid + 20, len(ctdset['meshes']))
            for index in xrange(first_index, last_index):
                rmesh, rroot = ctdset['meshes'][index], ctdset['trees'][index]
                if rmesh == lmesh:
                    continue
                rroots_minus.append(rroot)
                rsent = model.generate(rroot)
                rsents_minus.append(rsent)
        first_index = ctdset['meshes'].index(lmesh)
        last_index = first_index + len(rroots_plus)
        before_index = max(0, first_index - 10)
        after_index = min(last_index + 10, len(ctdset['meshes']))
        for index in xrange(before_index, after_index):
            if ctdset['meshes'][index] == lmesh:
                continue
            rroot = ctdset['trees'][index]
            rroots_minus.append(rroot)
            rsent = model.generate(rroot)
            rsents_minus.append(rsent)
        while len(rroots_minus) < 140:
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

        for j in xrange(5):
            plus_index = score_plus.index(max(score_plus))
            model.train(lroot, rroots_plus[plus_index], 5)
            minus_index = score_minus.index(max(score_minus))
            model.train(lroot, rroots_minus[minus_index], 1)
            del score_minus[minus_index]
            del rroots_minus[minus_index]
            
        del score_plus, score_minus, rsents_plus, rsents_minus, rroots_plus, rroots_minus
        gc.collect()

def evaluate_datasets(model, datasets, ctdset, output):
    rsents = []
    for rroot in ctdset['trees']:
        rsent = model.generate(rroot)
        rsents.append(rsent)

    for dataset in datasets:
        num_correct, num_pred = 0, 0
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
        output.write("\nno that mesh: %d\n" % no_that_mesh)
        output.write('cm_num_correct: %d\n' % cm_num_correct)
        output.write('cm_num_total: %d\n' % cm_num_total)
        output.write('num_pred_correct: %d\n' % num_correct)
        output.write('num_pred_total: %d\n' % num_pred)
        output.write('predict score: %f\n' % (float(num_correct)/float(num_pred)))
        output.write('total score: %f\n' % (float(num_correct+cm_num_correct)/float(num_pred+cm_num_total)))
    del rsents
    gc.collect()

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
    
    # old_model_path =  '../outputs/2016-11-17-10-43-23/2016-11-17-10-43-23.model'
    # utils.load_model(model, old_model_path)

    # old_ws_path = '../outputs/2016-11-17-10-43-23/2016-11-17-10-43-23.wstrain'
    # load_wrongsamples(test_set, old_ws_path)

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

    # evaluate_dataset(model, dev_set, ctd_set, output)
    # output.write('\n\n')
    # output.flush()
    
    for epoch in xrange(15):
        output.write('\nepoch: %d\n' % epoch)
        train_dataset(model, train_set, ctd_set)
        evaluate_datasets(model, [train_set, dev_set], ctd_set, output)
        output.flush()
    utils.save_model(model, model_path)
    output.write('\nevaluate on test set\n')
    evaluate_datasets(model, [test_set], ctd_set, output)
    for name in ['train', 'dev', 'test']:
        ws_name = curtime + '.ws' + name
        ws_path = os.path.join(output_dir, ws_name)
        save_wrongsamples(data[name], ws_path)

    output.close()

if __name__ == '__main__':
   	train_test()