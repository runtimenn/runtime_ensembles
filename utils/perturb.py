#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some helper functions to create adversarial images
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# own modules
import utils.adversary as adversary


#############################################################################
# adversarial methods (we use same params as in the Mahanalobis paper)


# fgst mahanal
def fgsm_attack(data, labels, model, criterion, random_noise_size, min_pixel, max_pixel, norm_trans_std, adv_noise):
    noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda()) 
    noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)
    # performs the simple fgsm attack
    # create torch variables
    inputs = Variable(data.data, requires_grad=True)
    # compute model output and predictions
    out = model(inputs)
    preds = out.argmax(dim=1, keepdim=True).detach()
    # calc loss
    loss = criterion(out, labels)
    # back - propagate, compute gradients
    loss.backward()
    # get grad of X
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float()-0.5)*2
    # the authors of Mahanalobis normalize the gradient for the cases of FGSM
    # and BIM. We apply the same normalization in order to keep the same setup
    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / norm_trans_std[0])
    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / norm_trans_std[1])
    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / norm_trans_std[2])

    adv_data = torch.add(inputs.data, adv_noise, gradient)  
    # clamp output
    adv_data = torch.clamp(adv_data, min_pixel, max_pixel)


    # get new predictions
    preds_new = model(adv_data).argmax(dim=1, keepdim=True).detach()
    # check which attacks were succesfull
    success = (preds.squeeze() != preds_new.squeeze()) & (preds.squeeze() == labels.squeeze())
    success = success.detach().cpu().numpy()
    # free cuda
    del inputs
    torch.cuda.empty_cache()
    # return
    return adv_data.detach(), success, preds, preds_new 
# end


#bim mahanal
def bim_attack(data, labels, model, criterion, random_noise_size, min_pixel, max_pixel, norm_trans_std, adv_noise):
    noisy_data = torch.add(data.data, random_noise_size, torch.randn(data.size()).cuda()) 
    noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)
    # performs the simple fgsm attack
    # create torch variables
    inputs = Variable(data.data, requires_grad=True)
    # compute model output and predictions
    out = model(inputs)
    preds = out.argmax(dim=1, keepdim=True).detach()
    # calc loss
    loss = criterion(out, labels)
    # back - propagate, compute gradients
    loss.backward()
    # get grad of X
    gradient = torch.sign(inputs.grad.data)
    
    for k in range(5): # do 5 iterations (as in paper)
        inputs = torch.add(inputs.data, adv_noise, gradient)
        inputs = torch.clamp(inputs, min_pixel, max_pixel)
        inputs = Variable(inputs, requires_grad=True)
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        gradient = torch.sign(inputs.grad.data)
        
        # normalize gradient (same as in FGSM)
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / norm_trans_std[0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / norm_trans_std[1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / norm_trans_std[2])

    adv_data = torch.add(inputs.data, adv_noise, gradient)
    # clamp output
    adv_data = torch.clamp(adv_data, min_pixel, max_pixel)

    # get new predictions
    preds_new = model(adv_data).argmax(dim=1, keepdim=True).detach()
    # check which attacks were succesfull
    success = (preds.squeeze() != preds_new.squeeze()) & (preds.squeeze() == labels.squeeze())
    success = success.detach().cpu().numpy()
    # free cuda
    del inputs
    torch.cuda.empty_cache()
    # return
    return adv_data.detach(), success, preds, preds_new 
# end


#deep fool mahanal
def df_attack(data, labels, model, n_classes, min_pixel, max_pixel, adv_noise):
    # get preds before attack
    out = model(data)
    preds = out.argmax(dim=1, keepdim=True).detach()
    
    # use fb attack
    _, adv_data = adversary.deepfool(model, data.data.clone(), labels.data.cpu(), 
                                             n_classes, step_size=adv_noise, train_mode=False)
    
    # clamp output
    adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
    
    # get new predictions
    preds_new = model(adv_data.cuda()).argmax(dim=1, keepdim=True).detach()
    # check which attacks were succesfull
    success = (preds.squeeze() != preds_new.squeeze()) & (preds.squeeze() == labels.squeeze())
    success = success.detach().cpu().numpy()
    # free cuda
    # del inputs
    # torch.cuda.empty_cache()
    # return
    return adv_data.detach(), success, preds, preds_new 
# end 


# cw mahanal
def cw_attack(data, labels, model, n_classes, min_pixel, max_pixel, adv_noise):
    # get preds before attack
    out = model(data)
    preds = out.argmax(dim=1, keepdim=True).detach()
    
    # use fb attack
    _, adv_data = adversary.cw(model, data.data.clone(), labels.data.cpu(), 1.0, 'l2', crop_frac=1.0)
    
    # get new predictions
    preds_new = model(adv_data).argmax(dim=1, keepdim=True).detach()
    # check which attacks were succesfull
    success = (preds.squeeze() != preds_new.squeeze()) & (preds.squeeze() == labels.squeeze())
    success = success.detach().cpu().numpy()
    # free cuda
    # del inputs
    # torch.cuda.empty_cache()
    # return
    return adv_data.detach(), success, preds, preds_new 
# end 



# mahanal all attacks
def adv_attk_all(adv_type, data, labels, n_classes, model, criterion, random_noise_size, min_pixel, max_pixel, norm_trans_std, adv_noise):
    if adv_type == 'FGSM':
        return fgsm_attack(data, labels, model, criterion, random_noise_size, min_pixel, max_pixel, norm_trans_std, adv_noise)
    elif adv_type == 'BIM':
        return bim_attack(data, labels, model, criterion, random_noise_size, min_pixel, max_pixel, norm_trans_std, adv_noise)
    elif adv_type == 'DeepFool':
        return df_attack(data, labels, model, n_classes, min_pixel, max_pixel, adv_noise)
    elif adv_type == 'CWL2':
        return cw_attack(data, labels, model, n_classes, min_pixel, max_pixel, adv_noise)
    else:
        # attack not understood
        return None, None, None, None
# end






