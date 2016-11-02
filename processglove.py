from vocab import Vocab
import numpy as np
import os

def read_embeddings_into_numpy(file_name, vocab=None):
    words, array = [], []
    with open(file_name, 'r') as f:
        for line in f:
            fields = line.strip().split()
            word = fields[0]
            if vocab and word not in vocab.word2idx:
                continue
            embeddings = np.array([float(field) for field in fields[1:]])
            words.append(word)
            array.append(embeddings)

    return np.array(words), np.array(array)

if __name__ == '__main__':
    vocab_path = '../data/ncbi/vocab-cased.txt'
    ncbi_dir = '../data/ncbi'
    glove_path = '../data/glove/glove.840B.300d.txt'
    vocab = Vocab()
    vocab.load(vocab_path)
    words, embeddings = read_embeddings_into_numpy(glove_path, vocab)
    
    np.save(os.path.join(ncbi_dir, 'words.npy'), words)
    np.save(os.path.join(ncbi_dir, 'glove.npy'), embeddings)
    
    print 'vocab size: ', vocab.size()
    print 'known words: ', len(words)
    print 'unknown words: ', (vocab.size() - len(words))
