import os
import glob
import random
import numpy as np
from vocab import Vocab
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import utils

def get_abbr(abbrpath):
    abbr = {}
    with open(abbrpath, 'r') as f:
        for line in f:
            pmid, short, full = line.strip().split('\t')
            if pmid not in abbr:
                abbr[pmid] = {}
            abbr[pmid][short] = full
    
    return abbr


def get_corpus(corpuspath):
    source = dict()
    with open(corpuspath, 'r') as corpus:
        corpus = corpus.read()
        abstracts = corpus.split('\r\n\r\n')
        for abstract in abstracts:
            title = abstract.split('\r\n')[0]
            diseases = abstract.split('\r\n')[2:]
            pmid = title.split('|')[0]
            name_mesh = set()
            for disease in diseases:
                dname = disease.split('\t')[3].strip()
                dmesh = disease.split('\t')[5]
                if dmesh.find('|') != -1 or dmesh.find('+') != -1:
                    continue
                name_mesh.add((dname, dmesh))
            source[pmid] = name_mesh
    return source

def get_pmids(pmidpath):
    pmids = []
    with open(pmidpath, 'r') as f:
        for line in f:
            pmids.append(line.strip())        
    return pmids

def write_nm(name, mesh, namef, meshf):
    if name.strip() == '':
        return 
    name = name.replace('-', ' ').replace('/', ' ').replace('&', ' ')
    parts = name.split()
    for i, part in enumerate(parts):
        if any(ch >= 'a' and ch <= 'z' for ch in part):
            parts[i] = part.lower()
    name = ' '.join(parts)
    namef.write(name + '\n')
    if mesh.find(':') != -1:
        meshf.write(mesh+'\n')
    else:
        meshf.write('MESH:'+mesh+'\n')

def create_ctd(ncbi_dir, cp):

    ctdpath = os.path.join(ncbi_dir, 'CTD_diseases-2015-06-04.tsv')
    ctddir = os.path.join(ncbi_dir, 'ctd')
    if not os.path.exists(ctddir):
        os.mkdir(ctddir)

    namepath = os.path.join(ctddir, 'name.txt')
    meshpath = os.path.join(ctddir, 'mesh.txt')

    with open(ctdpath, 'r') as ctd,\
        open(namepath, 'w') as ctdname,\
        open(meshpath, 'w') as ctdmesh:
        namelist, meshlist = [], []
        for i in xrange(28):
            ctd.readline()
        for line in ctd:
            content = line.split('\t')
            name = content[0].strip()
            mesh = content[1]
            syns = content[7].strip().split('|')
            write_nm(name, mesh, ctdname, ctdmesh)
            for syn in syns:
                write_nm(syn, mesh, ctdname, ctdmesh)
        ctdname.flush()
    dependency_parse(namepath, cp)

def create_corpus(ncbi_dir, cp):
    sname = ['train', 'dev', 'test']
    corpus = get_corpus(os.path.join(ncbi_dir, 'Corpus.txt'))
    abbr = get_abbr(os.path.join(ncbi_dir, 'abbreviations.tsv'))
    for cname in sname:
        pmidpath = os.path.join(ncbi_dir, 'NCBI_corpus_'+cname+'_PMIDs.txt')
        pmids = get_pmids(pmidpath)
        dir = os.path.join(ncbi_dir, cname)
        if not os.path.exists(dir):
            os.mkdir(dir)
        namepath = os.path.join(dir, 'name.txt')
        meshpath = os.path.join(dir, 'mesh.txt')
        dpmidpath = os.path.join(dir, 'pmid.txt')
        with open(namepath, 'w') as namef, open(meshpath, 'w') as meshf, open(dpmidpath,'w') as dpmidf:
            for pmid in pmids:
                if pmid not in corpus:
                    continue
                diseases = corpus[pmid]
                for disease in diseases:
                    dname = disease[0]
                    dmesh = disease[1]
                    parts = dname.split()
                    for i, part in enumerate(parts):
                        if pmid in abbr and part in abbr[pmid]:
                            part = abbr[pmid][part]
                            parts[i] = part
                    dname = ' '.join(parts).strip()
                    write_nm(dname, dmesh, namef, meshf)
                    dpmidf.write(pmid+'\n')
            namef.flush()
        dependency_parse(namepath, cp)
    
def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)

def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
        % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def stem(filepaths):
    dstpaths = [filepath+'s' for filepath in filepaths]
    stemmer = PorterStemmer()
    for filepath, dstpaths in zip(filepaths, dstpaths):
        with open(filepath, 'r') as inf, open(dstpaths, 'w') as ouf:
            for line in inf:
                words = line.strip().split()
                stem_words = [stemmer.stem(word.decode('utf-8')).encode('utf-8') for word in words]
                stem_line = ' '.join(stem_words)
                ouf.write(stem_line + '\n')

def lemmatize(filepaths):
    dstpaths = [filepath+'l' for filepath in filepaths]
    wnl = WordNetLemmatizer()
    for filepath, dstpaths in zip(filepaths, dstpaths):
        with open(filepath, 'r') as inf, open(dstpaths, 'w') as ouf:
            for line in inf:
                words = line.strip().split()
                stem_words = [wnl.lemmatize(word.decode('utf-8')).encode('utf-8') for word in words]
                stem_line = ' '.join(stem_words)
                ouf.write(stem_line + '\n')

def read_embeddings_into_numpy(ncbi_dir):
    vocab_path = os.path.join(ncbi_dir, 'vocab-cased.txt')
    glove_path = '../../data/glove.840B.300d.txt'

    vocab = Vocab()
    vocab.load(vocab_path)

    words, array = [], []
    with open(glove_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            word = fields[0]
            if vocab and word not in vocab.word2idx:
                continue
            embeddings = np.array([float(field) for field in fields[1:]])
            words.append(word)
            array.append(embeddings)

    np.save(os.path.join(ncbi_dir, 'words.npy'), words)
    np.save(os.path.join(ncbi_dir, 'glove.npy'), array)

    print 'vocab size: ', vocab.size()
    print 'known words: ', len(words)
    print 'unknown words: ', (vocab.size() - len(words)) 

def read_w2v_into_numpy(ncbi_dir):
    vocab_path = os.path.join(ncbi_dir, 'vocab-cased.txt')
    w2v_path = '../../data/stemmed_model.bin'

    vocab = Vocab()
    vocab.load(vocab_path)

    words, array = [], []
    with open(w2v_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab.word2idx:
               embedding = np.fromstring(f.read(binary_len), dtype='float32')
               words.append(word)
               array.append(embedding)  
            else:
                f.read(binary_len)
   
    np.save(os.path.join(ncbi_dir, 'words.npy'), words)
    np.save(os.path.join(ncbi_dir, 'glove.npy'), array)

    print 'vocab size: ', vocab.size()
    print 'known words: ', len(words)
    print 'unknown words: ', (vocab.size() - len(words)) 

if __name__ == '__main__':
    print '=' * 80
    print 'prerocessing ncbi corpus'
    print '=' * 80
    
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ncbi_dir = os.path.join(data_dir, 'ncbi')
    lib_dir = os.path.join(base_dir, 'lib')

    classpath = ':'.join([lib_dir,
            os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
            os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])
        
    create_ctd(ncbi_dir, classpath)
    create_corpus(ncbi_dir, classpath)

    stem(glob.glob(os.path.join(ncbi_dir, '*/*.toks')))

    # lemmatize(glob.glob(os.path.join(ncbi_dir, '*/*.toks')))
    build_vocab(
        glob.glob(os.path.join(ncbi_dir, '*/*.tokss')),
        os.path.join(ncbi_dir, 'vocab-cased.txt'),
        lowercase=False)  


    read_w2v_into_numpy(ncbi_dir)
    
    

