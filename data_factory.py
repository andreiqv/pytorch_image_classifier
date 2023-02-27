#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First run split_dataset.py to split dataset in train|valid|test parts.
Then run this script.
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader #, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

#import torchsample as ts

import settings

IMAGE_WIDTH = 224


data_transforms = {

    'train': transforms.Compose([
        transforms.Resize(256),
        #transforms.RandomResizedCrop(IMAGE_WIDTH),
        #transforms.RandomHorizontalFlip(),
        #transforms.Pad(padding=60, padding_mode='reflect'),
        #transforms.RandomRotation([0,90], expand=True),
        #transforms.RandomResizedCrop(IMAGE_WIDTH),
        #transforms.RandomRotation([-1,1], expand=True),
        transforms.RandomAffine(degrees=(-2, 2),translate=(0.01, 0.01)),
        transforms.CenterCrop(IMAGE_WIDTH),
        transforms.ColorJitter(0.1, 0.1, 0, 0),

        transforms.ToTensor(),
        #ts.transforms.Rotate(20), # data augmentation: rotation 
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ]),

    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_WIDTH),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ]),
}

def load_data(data_dir):
        
    data_parts = ['train', 'valid']

    image_datasets = {p: ImageFolder(os.path.join(data_dir, p),
        data_transforms[p]) for p in data_parts}

    dataloaders = {p: DataLoader(image_datasets[p], batch_size=settings.batch_size,
        shuffle=True, num_workers=settings.num_workers) for p in data_parts}

    return dataloaders, image_datasets


def dataset_info(image_datasets):

    data_parts = list(image_datasets.keys())
    print('data_parts:', data_parts)

    dataset_sizes = {p: len(image_datasets[p]) for p in data_parts}
    for p in data_parts:
        print('{0} size: {1}'.format(p, dataset_sizes[p]))

    class_names = image_datasets['train'].classes
    print('num_classes:', len(class_names))
    print('classes:', class_names)

    return dataset_sizes, class_names


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

#----------

if __name__ == '__main__':

    data_dir = settings.data_dir #'/w/WORK/ineru/06_scales/_dataset/splited/'
    dataloaders, image_datasets = load_data(data_dir)
    dataset_sizes, class_names = dataset_info(image_datasets)

    #for i, (x, y) in enumerate(dataloaders['valid']):
        #print(x) # image
    #    print(i, y) # image label

    # Get a batch of training data
    for inputs, classes in dataloaders['valid']:
        print(len(inputs))
        print(classes)
    
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])
        plt.pause(2)  # pause a bit so that plots are updated

        #for i, inp in enumerate(inputs):
        #    imshow(inp, title=classes[i])
        #    plt.pause(2)    