
class Vocab(object):
    
    def __init__(self):
        self.words = []
        self.word2idx = {}
        self.unk_index = None
        self.start_index = None
        self.end_index = None
        self.unk_token = None
        self.start_token = None 
        self.end_token = None
    
    def load(self, path):
        with open(path, 'r') as in_file:
            for line in in_file:
                word = line.strip()
                assert word not in self.word2idx
                self.word2idx[word] = len(self.words)
                self.words.append(word)
        
        for unk in ['<unk>', '<UNK>', 'UUUNKKK']:
            self.unk_index = self.unk_index or self.word2idx.get(unk, None)
            if self.unk_index is not None:
                self.unk_token = unk
                break

        for start in ['<s>', '<S>']:
            self.start_index = self.start_index or self.word2idx.get(start, None)
            if self.start_index is not None:
                self.start_token = start
                break
        
        for end in ['</s>', '</S>']:
            self.end_index = self.end_index or self.word2idx.get(end, None)
            if self.end_index is not None:
                self.end_token = end
                break
        
    
    def index(self, word):
        if self.unk_index is None:
            assert word in self.word2idx
        return self.word2idx.get(word, self.unk_index)
    
    def size(self):
        return len(self.words)
        
