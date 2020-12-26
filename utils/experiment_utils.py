# -*- coding: utf-8 -*-
"""
Some utilities used in the experiments
"""

# imports

import numpy as np
import tqdm

# torch
import torch
import torch.nn as nn

# own modules
import utils.perturb as perturb


#######################################################################
# collect layer activations
# methods to collect values of some hidden layer(s), on a dataset

# layer_sizes = [100, 100, 50, 10] # for example

def layer_ind(lay, layer_sizes, start_lay = 0):
    # get index of layer in patterns array
    # we store the activations of many layers in a contagious 2D array
    # this function returns the index of layer k in that array, assuming
    # we start counting from some start_layer
    
    low = high = 0 # indexes to return: from - to
    if lay < start_lay:
        return 0, 0
    # get lower index
    for i in range(lay):
        low += layer_sizes[i]
    # get upper index
    high = low + layer_sizes[lay]
    # subtract offset (if we start counting from a layer other than 0)
    offset = int( np.sum(layer_sizes[:start_lay]) )
    # return indexes
    return low - offset, high - offset

def collectActivMah(data_loader, model, layer_sizes, max_samples = 1e5, device = 'cuda'):
    # collects activation vectors for block layers and elements in dataset
    # for the denseNet
    # we follow the authors of Mahanalobis and average the intermediate
    # conv activations channel - wise
    
    # numb of data points and features
    n_data = len(data_loader.dataset)
    n_feat = sum(layer_sizes)
    
    A = np.zeros((n_data, n_feat)) # holds activ patterns for all layers
    Ares = np.zeros((n_data, 2), int) # holds (preds, labels) pair for all samples
    
    cnt = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output, feats = model.feature_list(data)
            pred = output.detach().cpu().argmax(dim=1, keepdim=False) # get the index of max
            n = len(pred)
            # for all layers, get intermediate outputs and trow data into monitor
            high = min(cnt + n, n_data)
            Ares[cnt:high, 0] = pred.numpy()[:(high - cnt)] 
            Ares[cnt:high, 1] = target.cpu().numpy()[:(high - cnt)] 
            for lay in range(len(layer_sizes)-1):
                # get the output of the layer to monitor
                intermediate = feats[lay]
                # add to A
                # get layer idx
                l, r = layer_ind(lay, layer_sizes)
                high = min(cnt + n, n_data)
                # get channel-wise mean (as in paper)
                A[cnt:high, l:r] = intermediate.detach().cpu().numpy().mean(axis = (2,3))[:(high - cnt)] 
            # add also the smax outputs
            # get layer idx
            l, r = layer_ind(len(layer_sizes) - 1, layer_sizes)
            A[cnt:high, l:r] = output.detach().cpu().numpy()[:(high - cnt)] 
            cnt = high
            if cnt > max_samples:
                break
    return A[:cnt], Ares[:cnt]
# end


def collectActivAdv(data_loader, model, n_classes, layer_sizes, adv_type, random_noise_size, 
                    min_pixel, max_pixel, norm_trans_std, adv_noise, max_samples = 1e5, device = 'cuda'):
    # collects activation vectors for all layers and elements in dataset
        
    criterion = nn.CrossEntropyLoss()
    n_data = len(data_loader.dataset)
    n_feat = sum(layer_sizes)
    A = np.zeros((n_data, n_feat)) # holds activ patterns for all layers
    Ares = np.zeros((n_data, 2), int) # holds (preds, labels)
    cnt = 0
    #with torch.no_grad():
    for data, target in tqdm.tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        
        # perform adversarial attack
        adv_data, success, _, _ = perturb.adv_attk_all(adv_type, data, target, n_classes, model, criterion, random_noise_size, min_pixel, max_pixel, norm_trans_std, adv_noise)
        
        # get succeeded adv samples 
        adv_data = adv_data[success]
        target = target[success]
        
        with torch.no_grad():
            output, feats = model.feature_list(adv_data.to(device))
            pred = output.detach().cpu().argmax(dim=1, keepdim=False) # get the index of max
            n = len(pred)
            # for all layers, get intermediate outputs and insert data into A
            high = min(cnt + n, n_data)
            Ares[cnt:high, 0] = pred.numpy()[:(high - cnt)] 
            Ares[cnt:high, 1] = target.cpu().numpy()[:(high - cnt)] 
            for lay in range(len(layer_sizes)-1):
                # get the output of the layer to monitor
                intermediate = feats[lay]
                # add to A
                # get layer idx
                l, r = layer_ind(lay, layer_sizes)
                high = min(cnt + n, n_data)
                A[cnt:high, l:r] = intermediate.detach().cpu().numpy().mean(axis = (2,3))[:(high - cnt)] 
            # add also the smax outputs
            # get layer idx
            l, r = layer_ind(len(layer_sizes) - 1, layer_sizes)
            A[cnt:high, l:r] = output.detach().cpu().numpy()[:(high - cnt)] 
            cnt = high
            if cnt > max_samples:
                break
    return A[:cnt], Ares[:cnt]
# end func


