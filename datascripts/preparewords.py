from nltk.stem.porter import PorterStemmer
import re

def get_ctdsentences(ctdpath, filepath):
    with open(ctdpath, 'r') as ctdf, open(filepath, 'w') as ouf :
        for i in xrange(28):
            ctdf.readline()
        for line in ctdf:
            content = line.split('\t')
            names = [content[0].strip()]
            names.extend(content[7].strip().split('|'))
            names.append('.')
            for i, name in enumerate(names):
                if name.strip() == '':
                    del names[i]
            names = ' , '.join(names)
            if names.strip() != '':
                ouf.write(names + '\n')
            definition = content[3]
            if definition.strip() != '':
                ouf.write(definition + '\n')

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    # string = re.sub(r"\'s", " 's", string) 
    # string = re.sub(r"\'ve", " 've", string) 
    # string = re.sub(r"n\'t", " n't", string) 
    # string = re.sub(r"\'re", " 're", string) 
    # string = re.sub(r"\'d", " 'd", string) 
    # string = re.sub(r"\'ll", " 'll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()    

def conv_sen_into_words(filepaths, dstfilepath):
    stemmer = PorterStemmer()
    for filepath in filepaths:
        with open(filepath, 'r') as inf, open(dstfilepath, 'a') as ouf:
            for line in inf:
                line = line.replace('-', ' ').replace('/', ' ').replace('&', ' ')
                parts = line.split()
                for i, part in enumerate(parts):
                    if any(ch > 'a' and ch < 'z' for ch in part):
                        parts[i] = part.lower()
                stem_parts = [stemmer.stem(part.decode('utf-8')).encode('utf-8') for part in parts]
                stem_line = ' '.join(stem_parts).strip()
                ouf.write(stem_line + '\n')


if __name__ == '__main__':
    # get_ctdsentences('../data/ncbi/CTD_diseases-2015-06-04.tsv', './ctd_sentences.txt')
    filepaths = ['../../../data/ctd_sentences.txt', '../../../data/Corpus_sentences.txt', '../../../data/gd_sentences.txt']
    dstfilepath = '../../../data/stemmed_words.txt'
    conv_sen_into_words(filepaths, dstfilepath)
            

