# -*- coding: utf-8 -*-
"""
Some DNN utilities (training and testing a model)
"""

# imports

# torch
import torch


# methods to train / test a model

# define functions to train and test the model
def train(model, criterion, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
# end train
    
def test(model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of max
            # sum up correctcly classified examples
            correct += pred.eq(target.view_as(pred)).sum().item()    
    test_loss /= len(test_loader.dataset)
    
    # print
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    accuracy = correct / len(test_loader.dataset)
    return accuracy
# end test


########################################################################
# some utils to set missing batch norm attributes

# older pytorch models lack some new attributes of the batch norm layer
# to fix that and load their weights, we set these attributes to some
# default values
# for more information, see: 
# https://github.com/ShiyuLiang/odin-pytorch/issues/3

def recursion_change_bn(module):
    # this function finds all batch layers in a model, and sets the
    # track running stats to a default value it had before
    # method from link above
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def fix_model(model):
    # fix all batch norm layers in an older model
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)

def load_pretrained_old(pre_trained_net, device = 'cuda'):
    # load an older pretrained model
    
    # load model to a given device
    # see: https://discuss.pytorch.org/t/how-to-convert-gpu-trained-model-on-cpu-model/63514/4
    model = torch.load(pre_trained_net, map_location=torch.device(device))

    # fix model
    fix_model(model)
    
    return model




