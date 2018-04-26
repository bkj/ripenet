#!/usr/bin/env python

"""
    data.py
"""

from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision
from basenet.vision import transforms as btransforms

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# def make_mnist_dataloaders(root='data', mode='mnist', train_size=1.0, train_batch_size=128, 
#     eval_batch_size=256, num_workers=8, seed=1111, download=False, pretensor=False, pin_memory=False):
    
#     if mode == 'mnist':
#         if pretensor:
#             transform = torchvision.transforms.Compose([
#                torchvision.transforms.Normalize((0.1307,), (0.3081,))
#             ])        
#         else:
#             transform = torchvision.transforms.Compose([
#                torchvision.transforms.ToTensor(),
#                torchvision.transforms.Normalize((0.1307,), (0.3081,))
#             ])
        
#         trainset = torchvision.datasets.MNIST(root='%s/mnist' % root, train=True, 
#             download=download, transform=transform, pretensor=pretensor)
#         testset = torchvision.datasets.MNIST(root='%s/mnist' % root, train=False, 
#             download=download, transform=transform, pretensor=pretensor)
    
#     elif mode == 'fashion_mnist':
#         if pretensor:
#             transform = None
#         else:
#             transform = torchvision.transforms.Compose([
#                torchvision.transforms.ToTensor(),
#             ])
        
#         trainset = torchvision.datasets.FashionMNIST(root='%s/fashion_mnist' % root, train=True, 
#             download=download, transform=transform, pretensor=pretensor)
#         testset = torchvision.datasets.FashionMNIST(root='%s/fashion_mnist' % root, train=False, 
#             download=download, transform=transform, pretensor=pretensor)
    
#     else:
#         raise Exception
    
#     return _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed, pin_memory)


def make_cifar_dataloaders(root='/home/bjohnson/projects/ripenet/data', mode='CIFAR10', train_size=1.0, train_batch_size=128, 
    eval_batch_size=128, num_workers=8, seed=1111, download=False, pin_memory=True, shuffle_train=True, shuffle_test=True):
    
    if mode == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            # <<
            # torchvision.transforms.RandomCrop(32, padding=4),
            # --
            btransforms.ReflectionPadding(margin=(4, 4)),
            torchvision.transforms.RandomCrop(32),
            # >>
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # <<
            # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # --
            btransforms.NormalizeDataset(dataset='cifar10'),
            # >>
        ])
        # !! `transform_train` gets applied to _val_ dataset as well
        
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # <<
            # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # --
            btransforms.NormalizeDataset(dataset='cifar10'),
            # >>
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='%s/CIFAR10' % root, train=True, download=download, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='%s/CIFAR10' % root, train=False, download=download, transform=transform_test)
    else:
        raise Exception
    
    return _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, 
        num_workers, seed, pin_memory, shuffle_train, shuffle_test)


def _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size,
    num_workers, seed, pin_memory, shuffle_train, shuffle_test):
    
    if train_size < 1:
        assert shuffle_train == True, 'data._make_loaders: incompatible arguments -- (train_size < 1) and (shuffle_train != True)'
        
        train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=train_size, random_state=seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
        )
        
        valloader = torch.utils.data.DataLoader(
            trainset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=pin_memory,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=pin_memory,
            shuffle=shuffle_train,
        )
        
        valloader = None
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=pin_memory,
        shuffle=shuffle_test,
    )
    
    return {
        "train" : trainloader,
        "test" : testloader,
        "val" : valloader,
    }
