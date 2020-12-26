# -*- coding: utf-8 -*-
"""
Some methods to measure performance (accuracy, precision, recall, AUC, etc.)

Note: most OOD papers use the following convention: a sample x is labelled
> True - 1 if it's known - in distribuiton (IND), and 
> False - if it's novel - out of distribution (OOD)
True positives, recalls, etc. are computed under this convention.
"""

# imports
import numpy as np


#####################################################################3
# methods

def scoresToClass(scores, thres):
    # converts scores to (binary) classes
    # if scores[i] <= thres -> ypred[i] = 0, 1 else
    
    ypred = scores > thres
    return ypred  

def getPrecRec(y, ypred):
    # get accuracy, precision, recall and F1 score
    # y are real labels (0-1), ypred predicted
    
    # true positives: we predict 1, and sample is really 1
    TP = np.sum( (y == 1) & (ypred == 1) )
    # false positives: we predict 1, but sample is 0
    FP = np.sum( (y == 0) & (ypred == 1) )
    # similarly 
    TN = np.sum( (y == 0) & (ypred == 0) )
    FN = np.sum( (y == 1) & (ypred == 0) )
    
    # get accuracy, precision, recall, and F1 score
    acc = np.sum( y == ypred ) / len(y)
    rec = TP / (TP + FN)
    prec = TP / (TP + FP)
    F1 = 2 * prec * rec / (prec + rec)
    
    return acc, prec, rec, F1


def printMeasures(Y_pred, Y_test):
    # print the measures of detect error and far
    # measure
    out_patt = np.sum(Y_pred)
    out_miss = np.sum(Y_pred & Y_test)
    real_miss = np.sum(Y_test)
    det_err = out_miss / real_miss
    far = 1 - out_miss / out_patt
    #print('detected error over total: {:.2f} %'.format( (det_err * 100) ))
    #print('false alarm rate: {:.2f} %'.format( (far * 100) ))
    tar = 1 - far
    F1 = 2 * det_err * tar / (det_err + tar)
    #print('F1 score: {:.2f} %'.format( (F1 * 100) ))
    return det_err, far, F1

def getTpFp(known, novel):
    # get  true positives and false positives
    # known: scores of known data
    # novel: scores on ood data
    # 0 <= scores <= 1, 0 means OOD, 1 IND
    # method adapted from the Mahanalobis paper
    
    # sort scores in ascending order
    known.sort()
    novel.sort()
    # get lengths
    num_k = len(known)
    num_n = len(novel)
    # init tp and fp arrays
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    
    # in the loop below, we create the tp and fp arrays for all thresholds
    # we compare novel with known scores and update the arrays accordingly
    
    tp[0], fp[0] = num_k, num_n # at threshold 0
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    return tp, fp

def calc_metrics(known, novel):
    # given the known and novel scores, calculate a 
    # series of metrics
    
    # get tp and fp
    tp, fp = getTpFp(known, novel)
    
    # fpr at 95% fpr
    tpr95_pos = np.abs(tp / len(known) - .95).argmin()
    tnr_at_tpr95 = 1 - fp[tpr95_pos] / len(novel)
    
    # auroc
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    auroc = -np.trapz(1.-fpr, tpr)
    
    # dtacc
    dtacc = .5 * (tp/tp[0] + 1 - fp/fp[0]).max()
    
    # auin
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    auin = -np.trapz(pin[pin_ind], tpr[pin_ind])
    
    # auout
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp)/denom, [.5]])
    auout = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
    
    return tnr_at_tpr95, auroc, dtacc, auin, auout
# end
