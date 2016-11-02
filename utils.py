import numpy as np
import theano
import os
from vocab import Vocab
from node import Node


theano.config.floatX = 'float32'


def _clear_indices(root):
    root.idx = None
    [_clear_indices(child) for child in root.children if child]

def _get_leaf_vals(root, with_labels=False):
    all_leaves = []
    layer = [root]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children):
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    vals, labels = [], []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        vals.append(leaf.val)
        if with_labels:
            labels.append(leaf.label)
    if with_labels:
        return vals, labels
    else:
        return vals

def _get_tree_traversal(root, start_idx=0, max_degree=None):
    if not root.children:
        tree = [-1] * max_degree
        tree.append(0)
        return [tree], [-1, root.val], []
    layers = []
    layer =[root]
    while layer:
        layers.append(layer[:])
        next_layer = []
        for node in layer:
            next_layer.extend([child for child in node.children if child])
        layer = next_layer
    
    tree, internal_vals, labels = [], [], []
    idx = start_idx
    for layer in reversed(layers):
        for node in layer:
            if node.idx is not None:
                assert all(child is None for child in node.children)
                continue
            child_idxs = [(child.idx if child else -1)
                         for child in node.children]
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_vals.append(node.val or -1)
            labels.append(node.label)
            idx += 1
    
    return tree, internal_vals, labels



def gen_inputs(root, max_degree=None, only_leaves_have_vals=False, with_labels=False):
    _clear_indices(root)
    if with_labels:
        x, leaf_labels = _get_leaf_vals(root, with_labels)
    else:
        x = _get_leaf_vals(root)
    tree, internal_x, internal_labels = _get_tree_traversal(root, len(x), max_degree)
    assert all(v is not None for v in x)
    if not only_leaves_have_vals:
        assert all(v is not None for v in internal_x)
        x.extend(internal_x)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'))


def _read_tree(parents, labels=None, constituency=False):
    nodes = {}
    parents = [p -1 for p in parents]
    for i in xrange(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = Node(val=idx)
                if prev is not None:
                    assert prev.val != node.val
                    node.add_child(prev)
                
                if labels is not None:
                    node.label = labels[idx]
                nodes[idx] = node

                parent = parents[idx]
                if parent in nodes:
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break
                
                prev = node
                idx = parent
    
    num_root = sum(node.parent is None for node in nodes.itervalues())
    assert num_root == 1

    if constituency:
        leaf_idx = 0
        for node in nodes.itervalues():
            if node.children:
                node.val = None
            else:
                node.val = leaf_idx
                leaf_idx += 1
    
    max_degree = max(len(node.children) for node in nodes.itervalues())
    
    return max_degree, root

def read_trees(parents_file, labels_file=None, constituency=False): #labels_file None for realateness model
    trees, max_degree = [], 0
    if labels_file is None:
        with open(parents_file, 'r') as f:
            for line in f:
                parents = [int(p) for p in line.strip().split()]
                degree, tree = _read_tree(parents, labels=None, constituency=False)
                max_degree = max(degree, max_degree)
                trees.append(tree)
    else:
        with open(parents_file, 'r') as pf, open(labels_file, 'r') as lf:
            while True:
                pline = pf.readline()
                lline = lf.readline() 
                if not pline or not lline:
                    break
                parents = [int(p) for p in pline.strip().split()]
                labels = [int(l) for l in lline.strip().split()]
                degree, tree = _read_tree(parents, labels, constituency)
                max_degree = max(degree, max_degree)
                trees.append(tree)

    return max_degree, trees

def read_sentences(path, vocab):
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            sentences.append([vocab.index(token) for token in tokens])

    return sentences

def map_tokens_labels(node, sentence, fine_grained=False):
    if node.val is not None:
        node.val = sentence[node.val]
    
    if node.label is not None:
        if fine_grained:
            node.label += 2
        else:
            if node.label < 0:
                node.label = 0
            elif node.label == 0:
                node.label = 1
            else:
                node.label = 2
    [map_tokens_labels(child, sentence, fine_grained) 
    for child in node.children if child]





