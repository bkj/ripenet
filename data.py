#!/usr/bin/env python

"""
    data.py
"""

from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def make_mnist_dataloaders(root='data', mode='mnist', train_size=1.0, train_batch_size=128, 
    eval_batch_size=256, num_workers=8, seed=1111, download=False, pretensor=False, pin_memory=False):
    
    if mode == 'mnist':
        if pretensor:
            transform = torchvision.transforms.Compose([
               torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])        
        else:
            transform = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        trainset = torchvision.datasets.MNIST(root='%s/mnist' % root, train=True, 
            download=download, transform=transform, pretensor=pretensor)
        testset = torchvision.datasets.MNIST(root='%s/mnist' % root, train=False, 
            download=download, transform=transform, pretensor=pretensor)
    
    elif mode == 'fashion_mnist':
        if pretensor:
            transform = None
        else:
            transform = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
            ])
        
        trainset = torchvision.datasets.FashionMNIST(root='%s/fashion_mnist' % root, train=True, 
            download=download, transform=transform, pretensor=pretensor)
        testset = torchvision.datasets.FashionMNIST(root='%s/fashion_mnist' % root, train=False, 
            download=download, transform=transform, pretensor=pretensor)
    
    else:
        raise Exception
    
    return _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed, pin_memory)


def make_cifar_dataloaders(root='data', mode='CIFAR10', train_size=1.0, train_batch_size=128, 
    eval_batch_size=128, num_workers=8, seed=1111, download=False, pin_memory=True):
    
    if mode == 'CIFAR10':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # !! `transform_train` gets applied to _val_ dataset as well
        
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='%s/CIFAR10' % root, train=True, download=download, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='%s/CIFAR10' % root, train=False, download=download, transform=transform_test)
    else:
        raise Exception
    
    return _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed, pin_memory)


def _make_loaders(trainset, testset, train_size, train_batch_size, eval_batch_size, num_workers, seed, pin_memory):
    if train_size < 1:
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
            shuffle=True,
        )
        
        valloader = None
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=pin_memory,
        shuffle=True,
    )
    
    return {
        "train" : trainloader,
        "test" : testloader,
        "val" : valloader,
    }
