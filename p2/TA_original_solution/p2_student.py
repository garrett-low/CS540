import numpy as np
from math import log2
import copy

'''
This script is using all 9 features (2,3,...,10) to create a tree, which serves as a template.
Todo: you need to modify this by using the several specified features to create your own tree 
Todo: you need to do the pruning yourself
Todo: you need to get all the output including the test results.
Todo: you also need to generate the tree of such the format in the writeup: 'if (x3 <= 6) return 2 .......'
'''

with open('breast-cancer-wisconsin.data', 'r') as f:
    a = [l.strip('\n').split(',') for l in f if '?' not in l]


a = np.array(a).astype(int)   # training data


def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * log2(p0) - p1 * log2(p1)


def infogain(data, fea, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)
    d1 = data[data[:, fea - 1] <= threshold]
    d2 = data[data[:, fea - 1] > threshold]
    if len(d1) == 0 or len(d2) == 0: return 0
    return entropy(data) - (len(d1) / count * entropy(d1) + len(d2) / count * entropy(d2))


def find_best_split(data):
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data)
    if c0 == c: return (2, None)
    if c0 == 0: return (4, None)
    ig = [[infogain(data, f, t) for t in range(1, 10)] for f in range(2, 11)]
    ig = np.array(ig)
    max_ig = max(max(i) for i in ig)
    if max_ig == 0:
        if c0 >= c - c0:
            return (2, None)
        else:
            return (4, None)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    fea, threshold = ind[0] + 2, ind[1] + 1
    return (fea, threshold)


def split(data, node):
    fea, threshold = node.fea, node.threshold
    d1 = data[data[:,fea-1] <= threshold]
    d2 = data[data[:, fea-1] > threshold]
    return (d1,d2)


class Node:
    def __init__(self, fea, threshold):
        self.fea = fea
        self.threshold = threshold
        self.left = None
        self.right = None



ig = [[infogain(a, fea, t) for t in range(1,10)] for fea in range(2,11)]
ig = np.array(ig)
ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
root = Node(ind[0] + 2, ind[1] + 1)


def create_tree(data, node):
    d1,d2 = split(data, node)
    f1, t1 = find_best_split(d1)
    f2, t2 = find_best_split(d2)
    if t1 == None: node.left = f1
    else:
        node.left = Node(f1,t1)
        create_tree(d1, node.left)
    if t2 == None: node.right = f2
    else:
        node.right = Node(f2,t2)
        create_tree(d2, node.right)

create_tree(a, root)

s1 = [root]
s2 = []
while s1:
    s2 = copy.deepcopy(s1)
    s1 = []
    for n in s2:
        if n != 2 and n != 4:
            print(n.fea, n.threshold)
            if n.left != None: s1 += [n.left]
            if n.right != None: s1 += [n.right]
        else:
            print(n)

    print()