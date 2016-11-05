import theano
from theano import tensor as T
import numpy as np
import treelstm
import utils

DIR = '../data/ncpi'
GLOVE_DIR = '../data/glove'
FINE_GRAINED = False
DEPENDENCY = True
SEED = 24
REG = 1e-4
BATCH_SIZE = 10
NUM_EPOCHS = 10
LEARNING_RATE = 0.05
EMB_DIM = 300
HIDDEN_DIM = 150
OUTPUT_DIM = 1
MAX_LABEL = 5 #label y in [1,K]
TRAINABLE_EMBEDDINGS = True

class RelatenessModel(treelstm.ChildSumTreeLSTM):
    
    def __init__(self, num_emb, max_degree, constituency):
        super(RelatenessModel, self).__init__(HIDDEN_DIM, 
            EMB_DIM, max_degree, LEARNING_RATE, constituency)
        self.num_emb = num_emb
        self.trainable_embeddings = TRAINABLE_EMBEDDINGS
        self.max_label = MAX_LABEL
        self.reg = REG

        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        if self.trainable_embeddings:
            self.params.append(self.embeddings)

        self.output_fn = self.create_output_fn()

        # self._batch_train = self._create_batch_train()
        self._train, self._predict, self._generate, self._score = self._create_train_and_predict() 

    def _create_train_and_predict(self):
        lx = T.ivector(name='lx')
        rx = T.ivector(name='rx')
        ltree = T.imatrix(name='ltree')
        rtree = T.imatrix(name='rtree')
        p = T.vector(name='p', dtype=theano.config.floatX)

        lemb_x = self.embeddings[lx] * T.neq(lx, -1).dimshuffle(0, 'x')
        remb_x = self.embeddings[rx] * T.neq(rx, -1).dimshuffle(0, 'x')

        
        # lsent = self.leaf_unit(self.embeddings[lx[0]])[0] if lx.shape[0] == 1 else self.compute_tree(lemb_x, ltree)
        # rsent = self.leaf_unit(self.embeddings[rx[0]])[0] if rx.shape[0] == 1 else self.compute_tree(remb_x, rtree)
        lsent = self.compute_tree(lemb_x, ltree)
        rsent = self.compute_tree(remb_x, rtree)

        pred_p = self.output_fn(lsent, rsent) 
        
        loss = self.kl_divergence(p, pred_p)
        updates = self.gradient_descent(loss)

        _train = theano.function([lx, rx, ltree, rtree, p] , [loss], updates=updates) 
        _predict = theano.function([lx, rx, ltree, rtree] , [pred_p])
        _generate = theano.function([lx, ltree], [lsent])
        _score = theano.function([lsent, rsent], [pred_p])

        return _train, _predict, _generate, _score
        
    def train(self, lroot, rroot, y):
        lx, ltree = utils.gen_inputs(lroot, self.degree, False)
        rx, rtree = utils.gen_inputs(rroot, self.degree, False)
        p = np.zeros([self.max_label], dtype=theano.config.floatX)
        if y == np.floor(y):
            p[int(np.floor(y)) - 1] = 1
        else:
            p[int(np.floor(y))] = y - np.floor(y)
            p[int(np.floor(y)) - 1] = np.floor(y) - y + 1
        # if lx.shape[0] > 1 and rx.shape[0] > 1:
        loss = self._train(lx, rx, ltree[:,:-1], rtree[:,:-1], p)[0]
        return loss
        # return 0, 0
    
    def predict(self, lroot, rroot):
        lx, ltree = utils.gen_inputs(lroot, self.degree, False)
        rx, rtree = utils.gen_inputs(rroot, self.degree, False)
        pred_p = self._predict(lx,  rx, ltree[:,:-1], rtree[:,:-1])[0]
        pred_y = np.dot(pred_p, np.arange(1,self.max_label + 1)) 
        return pred_y
    
    def generate(self, root):
        x, tree = utils.gen_inputs(root, self.degree, False)
        sent  = self._generate(x, tree[:,:-1])[0]
        return sent

    def getscore(self, lsent, rsent):
        pred_p = self._score(lsent, rsent)[0]
        pred_y = np.dot(pred_p, np.arange(1,self.max_label + 1))
        return pred_y

    def create_output_fn(self):
        self.W_x = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.W_a = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_p = theano.shared(self.init_matrix([self.max_label, self.hidden_dim]))
        self.b_p = theano.shared(self.init_vector([self.max_label]))
        self.params.extend([self.W_x, self.W_a, self.b_h, self.W_p, self.b_p])

        def fn(h_l, h_r):
            h_x = h_l * h_r
            h_a = h_l - h_r
            
            h_s = T.nnet.sigmoid(T.dot(self.W_x, h_x) + T.dot(self.W_a, h_a) + self.b_h)
            p = T.sum(T.nnet.softmax(T.dot(self.W_p, h_s) + self.b_p), axis=0)

            return p
        
        return fn
    
    def kl_divergence(self, p, pred_p):
        temp_p = p + T.eq(p, 0)
        return (p * T.log(temp_p / pred_p)).sum()
    
    def loss_with_l2_reg(self, kl, batch_size):
        return kl/batch_size + 0.5 * self.reg * sum(map(lambda  x: T.sqr(x).sum(), self.params))

