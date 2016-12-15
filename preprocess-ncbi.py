import os
import glob
import random
import shutil
import numpy as np
import Levenshtein
from vocab import Vocab
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from stop_words import get_stop_words
import utils




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





def create_ctd(ncbi_dir):

    ctdpath = os.path.join(ncbi_dir, 'CTD_diseases-2015-06-04.tsv')
    ctddir = os.path.join(ncbi_dir, 'ctd')
    if os.path.exists(ctddir):
        shutil.rmtree(ctddir)
    os.mkdir(ctddir)


    namepath = os.path.join(ctddir, 'name.txt')
    meshpath = os.path.join(ctddir, 'mesh.txt')
    alIDpath = os.path.join(ctddir, 'alid.txt')

    with open(ctdpath, 'r') as ctd,\
        open(namepath, 'w') as ctdname,\
        open(meshpath, 'w') as ctdmesh,\
        open(alIDpath, 'w') as ctdalid:
        namelist, meshlist = [], []
        for i in xrange(28):
            ctd.readline()
        for line in ctd:
            content = line.split('\t')
            names = [content[0].strip()]
            mesh = content[1].strip()
            altIDs = content[2].strip()
            syns = content[7].strip()
            if syns != '':
                names.extend(syns.split('|'))
            for name in names:
                ctdname.write(name + '\n')
                ctdmesh.write(mesh + '\n')
                ctdalid.write(altIDs + '\n')
        ctdname.flush()
        ctdmesh.flush()
        ctdalid.flush()

def create_corpus(ncbi_dir):
    sname = ['train', 'dev', 'test']
    corpus = get_corpus(os.path.join(ncbi_dir, 'Corpus.txt'))
    # 
    for cname in sname:
        pmidpath = os.path.join(ncbi_dir, 'NCBI_corpus_'+cname+'_PMIDs.txt')
        pmids = []
        with open(pmidpath, 'r') as f:
            for line in f:
                pmids.append(line.strip()) 
        
        subdir = os.path.join(ncbi_dir, cname)
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
        os.mkdir(subdir)

        namepath = os.path.join(subdir, 'name.txt')
        meshpath = os.path.join(subdir, 'mesh.txt')
        dpmidpath = os.path.join(subdir, 'pmid.txt')
        with open(namepath, 'w') as namef, open(meshpath, 'w') as meshf, open(dpmidpath,'w') as dpmidf:
            for pmid in pmids:
                if pmid not in corpus:
                    continue
                diseases = corpus[pmid]
                for disease in diseases:
                    dname = disease[0].strip()
                    dmesh = disease[1].strip()
                    namef.write(dname + '\n')
                    if ':' not in dmesh:
                        dmesh = 'MESH:' + dmesh
                    meshf.write(dmesh + '\n')
                    dpmidf.write(pmid+'\n')
            namef.flush()
            meshf.flush()
            dpmidf.flush()

def resolve_abbr(ncbi_dir):
    abbrpath = os.path.join(ncbi_dir, 'abbreviations.tsv')
    abbr = {}
    with open(abbrpath, 'r') as f:
        for line in f:
            pmid, short, full = line.strip().split('\t')
            if pmid not in abbr:
                abbr[pmid] = {}
            abbr[pmid][short] = full
    
    dirnames = ['train', 'dev', 'test']
    for dirname in dirnames:
        dirpath = os.path.join(ncbi_dir, dirname)
        namepath = os.path.join(dirpath, 'name.txt')
        pmidpath = os.path.join(dirpath, 'pmid.txt')
        newnamepath = os.path.join(dirpath, 'name.full')
        with open(namepath, 'r') as namef,\
            open(pmidpath, 'r') as pmidf,\
            open(newnamepath, 'w') as newnamef:
            for line, pmid in zip(namef, pmidf):
                pmid = pmid.strip()
                if pmid not in abbr.keys():
                    newnamef.write(line)
                    continue
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part in abbr[pmid].keys():
                        parts[i] = abbr[pmid][part]
                newname = ' '.join(parts).strip()
                newnamef.write(newname + '\n')
            newnamef.flush()
    return

# remove some special characters, replace Roman numerals with Arabic numerals, 
# formmat character case, stem, remove stop words
def tokenize_phrase(ncbi_dir):
    r2a_map = {'I':'1', 'II':'2', 'III':'3', 'IV':'4'}
    stop_words = get_stop_words('en')
    stemmer = PorterStemmer()

    ctdpath = os.path.join(ncbi_dir, 'ctd')
    ctdnamepath = os.path.join(ctdpath, 'name.txt')
    ctdtokenpath = os.path.join(ctdpath, 'name.tokss')
    namepaths, tokenpaths = [ctdnamepath], [ctdtokenpath]
    dirnames = ['train', 'dev', 'test']
    for dirname in dirnames:
        dirpath = os.path.join(ncbi_dir, dirname)
        namepath = os.path.join(dirpath, 'name.full')
        tokenpath = os.path.join(dirpath, 'name.toks')
        namepaths.append(namepath)
        tokenpaths.append(tokenpath)
    
    for namepath, tokenpath in zip(namepaths, tokenpaths):
        with open(namepath, 'r') as namef,\
            open(tokenpath, 'w') as tokenf:
            for phrase in namef:
                phrase = phrase.strip().replace('-', ' ').replace('/', ' ')
                phrase = phrase.replace('&', ' ').replace(',', ' ')
                phrase = phrase.replace('(', ' ').replace(')', ' ')
                phrase = phrase.replace(';', ' ')
                parts = phrase.split()
                for i, part in enumerate(parts):
                    if part in r2a_map.keys():
                        part = r2a_map[part]
                    if any(ch >= 'a' and ch <= 'z' for ch in part):
                        part = part.lower()
                    if len(part) > 3 and (part[-1] >= '0' and part[-1] <= '9') \
                        and ((part[0] >= 'a' and part[0] <= 'z') or \
                            (part[0] >= 'A' and part[0] <= 'Z')):
                        for j, ch in enumerate(part):
                            if ch >= '0' and ch <= '9':
                                break
                        part = part[:j] + ' ' + part[j:]
                        # print part
                    parts[i] = stemmer.stem(part.decode('utf-8')).encode('utf-8')
                new_parts = [part.strip() for part in parts \
                            if part.decode('utf-8') not in stop_words]
                phrase = ' '.join(parts).strip() if len(new_parts) < 1 else ' '.join(new_parts).strip()
                tokenf.write(phrase + '\n')
            tokenf.flush()
    return 

def resolve_ambiguity(ncbi_dir):
    ctdpath = os.path.join(ncbi_dir, 'ctd')
    ctdtokenpath = os.path.join(ctdpath, 'name.tokss')
    ctd_vocab = set()
    with open(ctdtokenpath, 'r') as ctdtokenf:
        for line in ctdtokenf:
            words = line.strip().split()
            ctd_vocab.update(words)
    
    dirnames = ['train', 'dev', 'test']
    for dirname in dirnames:
        dirpath = os.path.join(ncbi_dir, dirname)
        tokepath = os.path.join(dirpath, 'name.toks')
        newtokenpath = os.path.join(dirpath, 'name.tokss')
        with open(tokepath, 'r') as tokenf,\
            open(newtokenpath, 'w') as newtokenf:
            for line in tokenf:
                tokens = line.strip().split()
                for i, token in enumerate(tokens):
                    if token in ctd_vocab or len(token) < 5 or \
                    not any(ch >= 'a' and ch <= 'z'for ch in token):
                        continue
                    candidate_tokens = [word for word in ctd_vocab \
                                        if Levenshtein.distance(word, token) < 2]
                    if len(candidate_tokens) > 0:
                        # print token, candidate_tokens
                        tokens[i] = candidate_tokens[0]

                newline = ' '.join(tokens).strip()
                newtokenf.write(newline + '\n')
            newtokenf.flush()
    return

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

def grammar_parse(ncbi_dir, cp):
    dirnames = ['train', 'dev', 'test', 'ctd']
    for dirname in dirnames:
        dirpath = os.path.join(ncbi_dir, dirname)
        tokenpath = os.path.join(dirpath, 'name.tokss')
        dependency_parse(tokenpath, cp)

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
        
    create_ctd(ncbi_dir)
    create_corpus(ncbi_dir)
    
    # abbreviatons resolve
    # formmat character case, raplace Roman numerals with Arabic numerals, remove stop words, stem
    # formmat some words that show in Corpus and not in CTD(resolve ambiguity)
    resolve_abbr(ncbi_dir)
    tokenize_phrase(ncbi_dir)
    resolve_ambiguity(ncbi_dir)
    grammar_parse(ncbi_dir, classpath)

    build_vocab(
        glob.glob(os.path.join(ncbi_dir, '*/*.toks')),
        os.path.join(ncbi_dir, 'vocab-cased.txt'),
        lowercase=False)  


    read_w2v_into_numpy(ncbi_dir)
    
    

