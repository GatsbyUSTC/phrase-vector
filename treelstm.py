import theano
import numpy as np
from theano.compat.python2x import OrderedDict
from theano import tensor as T

class ChildSumTreeLSTM(object):
    def __init__(self, hidden_dim, emb_dim, degree, learning_rate, constituency):
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.constituency = constituency
        self.params = []

        self.leaf_unit = self.create_leaf_unit()
        self.recursive_unit = self.create_recursive_unit()


    def create_recursive_unit(self):
        self.W_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_i = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_f = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_o = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_u = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_o, self.b_o,
        ])

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis = 0)
            i = T.nnet.sigmoid(T.dot(self.W_i, parent_x) + T.dot(self.U_i, h_tilde) + self.b_i)
            o = T.nnet.sigmoid(T.dot(self.W_o, parent_x) + T.dot(self.U_o, h_tilde) + self.b_o)
            u = T.tanh(T.dot(self.W_u, parent_x) + T.dot(self.U_u, h_tilde) + self.b_u)
            f = (T.nnet.sigmoid(T.dot(self.W_f, parent_x).dimshuffle('x',0) +\
                 T.dot(child_h, self.U_f.T) + self.b_f.dimshuffle('x',0))) *\
                 child_exists.dimshuffle(0,'x')
            c = i * u + T.sum(f * child_c, axis = 0)
            h = o * T.tanh(c)
            return h, c
        
        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            return self.recursive_unit(leaf_x, dummy, dummy, dummy.sum(axis = 1))

        return unit
    
    def compute_tree(self, emb_x, tree):
        num_nodes = tree.shape[0] #internal nodes
        num_leaves = emb_x.shape[0] - num_nodes #leaves

        (leaf_h, leaf_c), _ = theano.map(
            fn = self.leaf_unit,
            sequences = [emb_x[:num_leaves]]) #change

        if self.constituency:
            init_node_h = leaf_h
            init_node_c = leaf_c
            shift = 0
        else:
            dummy_internal_h = T.zeros([num_nodes, self.hidden_dim], dtype=theano.config.floatX)#theano.shared(self.init_vector([num_nodes, self.hidden_dim]))
            dummy_internal_c = T.zeros([num_nodes, self.hidden_dim], dtype=theano.config.floatX)#theano.shared(self.init_vector([num_nodes, self.hidden_dim]))
            init_node_h = T.concatenate([dummy_internal_h, leaf_h])
            init_node_c = T.concatenate([dummy_internal_c, leaf_c])
            shift = 1

        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h):
            child_exists = node_info > -1
            offset =  num_nodes * shift - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            node_h = T.concatenate([node_h, parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c, parent_c.reshape([1, self.hidden_dim])])

            return node_h[1:], node_c[1:], parent_h
        
        dummy_parent_h = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            fn = _recurrence,
            outputs_info=[init_node_h, init_node_c, dummy_parent_h],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return parent_h[-1]

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)
    
    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)
            
    def gradient_descent(self, loss):
        grad = T.grad(loss, self.params)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param, grad * (5.0 /scaling_den))
            updates[param] = param - self.learning_rate * grad

        return updates
            

