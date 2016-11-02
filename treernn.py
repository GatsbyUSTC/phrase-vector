import numpy as np
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict

class TreeRNN(object):
    def __init__(self, num_emb, emb_dim, hidden_dim, output_dim,
                degree=2, learning_rate= 0.01, momentum=0.9,
                trainable_embeddings=True):
        assert emb_dim > 1 and hidden_dim > 1
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.trainable_embeddings = trainable_embeddings

        self.params = []
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        if trainable_embeddings:
            self.params.append(self.embeddings)
        
        self.x = theano.tensor.ivector(name = 'x')
        self.tree = theano.tensor.imatrix(name = 'tree')
        self.y = theano.tensor.fvector(name = 'y')

        self.num_words = self.x.shape[0] #total number of nodes(leaves + internal) in tree
        emb_x = self.embeddings[self.x]
        emb_x = emb_x * theano.tensor.neq(self.x, -1).dimshuffle(0, 'x')

        self.tree_states = self.compute_tree(emb_x, self.tree)
        self.final_state = self.tree_states[-1]

        self.output_fn = self.create_output_fn()
        self.pred_y = self.output_fn(self.final_state)
        self.loss = self.loss_fn(self.y, self.pred_y)

        updates = self.





    def init_matrix(self, shape):
        return np.random.normal(scale=0.1,size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype = theano.config.floatX)

    def create_recursive_unit(self):
        self.W_hx = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.W_hh = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([self.W_hx, self.W_hh, self.b_h])
        def unit(parent_x, child_h):
            h_tilde = T.sum(child_h, axis=0)
            h = T.tanh(self.b_h + T.dot(self.W_hx, parent_x) + T.dot(self.W_hh, h_tilde))
            return h
        return unit 

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_matrix(self.degree, self.hidden_dim))
        def unit(leaf_x):
            return self.create_recursive_unit(leaf_x, dummy)
        return unit
    
    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])
        
        def fn(final_state):
            return T.nnet.softmax(
                T.dot(self.W_out, final_state) + self.b_out)
        
        return fn

    def compute_tree(self, emb_x, tree):
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_nodes= tree.shape[0]
        num_leaves = self.num_words - num_nodes

        leaf_h,_ = theano.map(fn = leaf_unit, sequences = [emb_x[:num_leaves]])
        init_node_h = leaf_h

        #size of node_h is unchangable, 
        #each time discard one node hidden state and store new hidden state
        #so each time we need compute a offset on top of node info
        def _recurrence(cur_emb, node_info, t, node_h, last_h):
            child_exists = node_info > -1
            offset = - child_exits * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h = self.recursive_unit(cur_emb,child_h)
            node_h = T.concatenate([node_h, parent_h.reshape([1, self.hidden_dim])])

            return node_h[1:], parent_h
        
        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, parent_h), _ = theano.scan(
            fn = _recurence,
            outputs_info=[init_node_h, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arrange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h,parent_h], axis = 0)

    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    def gradient_descent(self, loss):
        grad = T.grad(loss, self.params)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda  x:T.sqrt(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den  = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0/scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates