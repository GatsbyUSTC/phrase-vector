import numpy as np
import theano
import theano.tensor as T
# import utils
# from relateness import RelatenessModel
# from vocab import Vocab
import os

ncbi_dir = '../../data/ncbi'

# vocab = {}
# for name in names:
#     vocab[name] = set()
#     path = os.path.join(ncbi_dir, name)
#     path = os.path.join(path, 'name.toks')
#     with open(path, 'r') as f:
#         for line in f:
#             vocab[name] |= set(line.lower().split())
# in_count, outcount, total = 0, 0, 0
# for word in vocab['dev']:
#     if word in vocab['ctd']:
#         in_count += 1
#     else:
#         outcount += 1
#     total += 1
# print in_count, outcount, total
dirnames = ['train', 'dev', 'ctd']
known_dicts = {}
for dirname in dirnames:
    dirpath = os.path.join(ncbi_dir, dirname)
    tokenpath = os.path.join(dirpath, 'name.tokss')
    idpath = os.path.join(dirpath, 'mesh.txt')
    with open(tokenpath, 'r') as tokenf,\
        open(idpath, 'r') as idf:
        for token, meshid in zip(tokenf, idf):
            if token in known_dicts.keys():
                known_dicts[token].add(meshid.strip())
            else:
                known_dicts[token] = set([meshid.strip()])

dirname = 'ctd'
dirpath = os.path.join(ncbi_dir, dirname)
tokenpath = os.path.join(dirpath, 'name.tokss')
alidpath = os.path.join(dirpath, 'alid.txt')
with open(tokenpath, 'r') as tokenf,\
    open(alidpath, 'r') as alidf:
    for token, omids in zip(tokenf, alidf):
        if omids.strip() == '':
            continue
        omids = omids.strip().split('|')
        known_dicts[token].update(omids)

dirpath = os.path.join(ncbi_dir, 'test')
tokenpath = os.path.join(dirpath, 'name.tokss')
idpath = os.path.join(dirpath, 'mesh.txt')
total_num, in_num, right_num = 0, 0, 0
with open(tokenpath, 'r') as tokenf,\
    open(idpath, 'r') as idf:
    for i, (token, meshid) in enumerate(zip(tokenf, idf)):
        total_num += 1
        if token in known_dicts.keys():
            in_num += 1
            if meshid.strip() in known_dicts[token]:
                right_num += 1
            else:
                print meshid.strip() , token.strip(),
                print i, known_dicts[token]
print total_num, in_num, right_num
print in_num/float(total_num)
print right_num/float(total_num)