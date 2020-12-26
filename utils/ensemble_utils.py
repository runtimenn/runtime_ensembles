# -*- coding: utf-8 -*-
"""
Some utiities to build simple ensembles
"""

# imports

import copy
import numpy as np
#import matplotlib.pyplot as plt
import tqdm

# custom modules
import utils.experiment_utils as exp_utils


##########################################################################
# one vs k ensemble utils

def train_base_clf(classifier, A, Ares, lay, layer_sizes, portion):
    # train a classifier, taking DNN layer lay as input
    
    # get classifier
    clf = copy.deepcopy(classifier)
    
    # get indexes of given layers
    lay_ind1, lay_ind2 = exp_utils.layer_ind(lay, layer_sizes)
    
    # get random portion of dataset
    n_data = len(A)
    idx = np.random.choice(np.where(np.ones(n_data, int) == 1)[0], int(n_data * portion))
    
    # make training and test sets
    Xtrain = A[idx, lay_ind1 : lay_ind2]
    Ytrain = Ares[idx, 1]
    # rest as validation
    Xval = A[~idx, lay_ind1 : lay_ind2]
    Yval = Ares[~idx, 1]
    
    # train it
    clf.fit(Xtrain, Ytrain)
    
    # evaluate
    acc = clf.score(Xval, Yval)
    
    # return trained classifier
    return clf, acc

def test_base_clf(A, Ares, clf, lay, layer_sizes):
    # test a classifier on the given set B, Bres
    
    # get indexes of given layers
    lay_ind1, lay_ind2 = exp_utils.layer_ind(lay, layer_sizes)
    
    # get val set
    Xval = A[:, lay_ind1 : lay_ind2]
    #Yval = Bres[:, 0] == Bres[:, 1]
    
    # get preds
    Ypred = clf.predict(Xval)
    
    # get probs
    Ppred = clf.predict_proba(Xval)
    
    # return results
    return Ypred, Ppred

def train_all_classifiers_clf(classifier, A, Ares, n_classes, lay, layer_sizes, n_classifiers, portion):
    # train k classifiers
    
    classifiers_list = []
    
    for i in tqdm.tqdm(range(n_classifiers)):
        clf, _ = train_base_clf(classifier, A, Ares, lay, layer_sizes, portion)
        classifiers_list.append(clf)
    # end for
    
    # ready
    return classifiers_list

def test_all_classifiers_clf(classifiers_list, A1, A1res, A2, A2res, n_classes, lay, layer_sizes):
    # test all clfs for IND and OOD
    # return 2 arrays, with probs on IND, and OOD, from all clfs
    # and 2 more arrays with Ypred for IND and OOD, for all clfs
    
    # init the arrays
    n_classifiers = len(classifiers_list)
    # for IND
    Ypred_ind = np.zeros( (len(A1), n_classifiers), int )
    Ppred_ind = np.zeros( (len(A1), n_classes, n_classifiers) )
    
    # for OOD
    Ypred_ood = np.zeros( (len(A2), n_classifiers), int )
    Ppred_ood = np.zeros( (len(A2), n_classes, n_classifiers) )
    
    # for all classifiers
    for i in tqdm.tqdm( range(n_classifiers) ):
        
        # get classifier
        clf = classifiers_list[i]
        
        # get IND results of that classifier
        Ypred_clf, Ppred_clf = test_base_clf(A1, A1res, clf, lay, layer_sizes)
        # put them into arrays
        Ypred_ind[:, i] = Ypred_clf.copy()
        Ppred_ind[:, :, i] = Ppred_clf.copy()
        
        # get OOD results of that classifier
        Ypred_clf, Ppred_clf = test_base_clf(A2, A2res, clf, lay, layer_sizes)
        # put em into arrays
        Ypred_ood[:, i] = Ypred_clf.copy()
        Ppred_ood[:, :, i] = Ppred_clf.copy()
    # end for
    
    # ready
    return Ypred_ind, Ppred_ind, Ypred_ood, Ppred_ood


def do_all_clf(classifier, A1, A1res, A2, A2res, A3, A3res, n_classes, lay, layer_sizes, n_classifiers, portion):
    # train and test all classifiers
    classifiers_list = train_all_classifiers_clf(classifier, A1, A1res, n_classes, lay, layer_sizes, n_classifiers, portion)
    # test all classifiers
    return test_all_classifiers_clf(classifiers_list, A1, A1res, A3, A3res, n_classes, lay, layer_sizes)
# end


#######################################################################
# methods to build a regression ensemble

def train_base_reg(classifier, A, Ares, lay, layer_sizes, portion):
    # trains a base classifier, taking input from layer from lay
    # portion: percent of the data to take randomly
    
    # get classifier
    clf = copy.deepcopy(classifier)
    
    # get indexes of given layers
    lay_ind1, lay_ind2 = exp_utils.layer_ind(lay, layer_sizes)
    
    # get random portion of dataset
    n_data = len(A)
    idx = np.random.choice(np.where(np.ones(n_data, int) == 1)[0], int(n_data * portion))
    
    # make training and test sets
    Xtrain = A[idx, lay_ind1 : lay_ind2]
    Ytrain = Ares[idx]
    # rest as validation
    Xval = A[~idx, lay_ind1 : lay_ind2]
    Yval = Ares[~idx]
    
    # train it
    clf.fit(Xtrain, Ytrain)
    
    # evaluate
    acc = clf.score(Xval, Yval)
    
    # return
    return clf, acc

def test_base_reg(A, Ares, clf, lay, layer_sizes):
    # test a classifier on the given set B, Bres
    
    # get indexes of given layers
    lay_ind1, lay_ind2 = exp_utils.layer_ind(lay, layer_sizes)
    
    # get val set
    Xval = A[:, lay_ind1 : lay_ind2]
    #Yval = Bres[:, 0] == Bres[:, 1]
    
    # get preds
    Ypred = clf.predict(Xval)
    
    # get probs
    #Ppred = clf.predict_proba(Xval)
    
    # return
    return Ypred

def train_all_classifiers_reg(n_classifiers, classifier, A, Ares, lay, layer_sizes, portion):
    # train a list of classifiers
    
    classifiers_list = []
    
    for i in tqdm.tqdm(range(n_classifiers)):
        clf, _ = train_base_reg(classifier, A, Ares, lay, layer_sizes, portion)
        classifiers_list.append(clf)
    
    # ready
    return classifiers_list

def test_all_classifiers_reg(classifiers_list, A1, A1res, A2, A2res, n_classes, lay, layer_sizes):
    # test all classifiers for IND and OOD
    # return 2 arrays, with probs on IND, and OOD, from all classifiers
    # and 2 more arrays with Ypred for IND and OOD
    
    # init the arrays
    n_classifiers = len(classifiers_list)
    # for IND
    Ypred_ind = np.zeros( (len(A1), n_classes, n_classifiers))
    
    # for OOD
    Ypred_ood = np.zeros( (len(A2), n_classes, n_classifiers))
    
    # for all classes
    for i in tqdm.tqdm( range(n_classifiers) ):
        
        # get classifier
        clf = classifiers_list[i]
        
        # get IND results of that classifier
        Ypred_clf = test_base_reg(A1, A1res, clf, lay, layer_sizes)
        # put em into arrays
        Ypred_ind[:, :, i] = Ypred_clf.copy()
        
        # get OOD results of that classifier
        Ypred_clf = test_base_reg(A2, A2res, clf, lay, layer_sizes)
        # put em into arrays
        Ypred_ood[:, :, i] = Ypred_clf.copy()    
    # return all
    return Ypred_ind, Ypred_ood

# end


#######################################################################
# performance calculations

def entropy_scores(Ares, Ppred):
    # get entropy score of each sample
    # from the ensemble probability predictions, Ppred
    
    n_data = len(Ares)
    n_classes = Ppred.shape[1]
    n_clfs = Ppred.shape[2]
    
    # array to hold prediciton confidences
    entr_scores = np.zeros(n_data)
    x = np.zeros(n_clfs) # array for samples
    
    max_entr = n_classes * np.log2(n_classes)
    
    for i in range(n_data):
        # get DNN pred
        dnn_pred_cl = Ares[i, 0]
        # get ensemble probas for predicted class
        x = Ppred[i, dnn_pred_cl, :]
        
        # calc entropy
        idx = x > 0
        entr_scores[i] = - np.sum(x[idx] * np.log2(x[idx]))
        # for technical reasons, we subtract from max entropy,
        # because metrics expect known samples to have high score
        # and novel low - this is just a linear shift
        entr_scores[i] = max_entr - entr_scores[i]
    return entr_scores
