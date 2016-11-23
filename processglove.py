from vocab import Vocab
import numpy as np
import os

def read_embeddings_into_numpy():
    vocab_path = '../data/ncbi/vocab-cased.txt'
    glove_path = '../data/glove/glove.840B.300d.txt'
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
    np.save(os.path.join(ncbi_dir, 'glove.npy'), embeddings)
    
    print 'vocab size: ', vocab.size()
    print 'known words: ', len(words)
    print 'unknown words: ', (vocab.size() - len(words))

def read_w2v_into_numpy():
    vocab_path = '../data/ncbi/vocab-cased.txt'
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
   
    # np.save(os.path.join(ncbi_dir, 'words.npy'), words)
    # np.save(os.path.join(ncbi_dir, 'glove.npy'), array)

    for word in vocab.word2idx:
        if word not in words:
            print word

    print 'vocab size: ', vocab.size()
    print 'known words: ', len(words)
    print 'unknown words: ', (vocab.size() - len(words)) 

if __name__ == '__main__':
    read_w2v_into_numpy()
    

