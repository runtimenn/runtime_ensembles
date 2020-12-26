# -*- coding: utf-8 -*-
"""
Some utilities to run the method of Cheng et al.
"""

import numpy as np
import tqdm

# methods

def Patt(X):
    # transforms layer activations into 0-1 patterns
    # e.g., neuron is active or not
    return (X > 0)

def minHamdist(x, A):
    # get the minimum Hamming dist between vector x and 
    # the row vectors of array A
    return np.min(np.sum(x != A, axis = 1))
 
def calcHammDists(A, Ares, B, Bres, layer_sizes):
    # calculates hamming dists for all patterns in the test set B
    # with respect to training patterns in A
    # Ares contains rows of [y_pred, y_real], where y_pred is the
    # predicted class, etc. Similar for Bres
        
    n_data = len(B)
    # min Hamming dists array
    dists = np.zeros(n_data, int)
    # get last hidden layer
    # get indexes of that layer in A, B
    r = sum(layer_sizes) - layer_sizes[-1]
    l = r - layer_sizes[-2]
    
    # for all test points
    for i in tqdm.tqdm( range(n_data) ):
        # get x and it's predicted class
        x = B[i, l:r]
        c = Bres[i, 0]
        # get correctly classified samples of class c in training set A
        # Cheng et al. contract patterns using only the correct samples
        # predicted[i] = label[i] = c
        idx = (Ares[:, 0] == c) & (Ares[:, 0] == Ares[:, 1])
        # get hamming dist of x with these samples
        dists[i] = minHamdist(x, A[idx, l:r])
    return dists





