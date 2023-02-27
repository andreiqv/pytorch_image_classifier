#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First run split_dataset.py to split dataset in train|valid|test parts.
Then run this script.

train loss=2.6459, acc=0.3861, top1=0.3861, top6=0.4985
valid loss=1.4961, acc=0.5917, top1=0.5847, top6=0.7120
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from torch.autograd import Variable

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy

import progressbar
from PIL import Image, ImageDraw, ImageFont

import data_factory
from data_factory import data_transforms
import settings
from settings import data_dir, TOPk, SHOW_BAR, DEBUG
from accuracy import *
#SHOW_BAR = True
#DEBUG = False
#TOPk = 6

#root = '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018/'
#data_dir = '/w/WORK/ineru/06_scales/_dataset/splited/'

dataloaders, image_datasets = data_factory.load_data(data_dir)
#data_parts = list(dataloaders.keys())
dataset_sizes, class_names = data_factory.dataset_info(image_datasets)
num_classes = len(class_names)
data_parts = ['train', 'valid']
class_to_idx = image_datasets['train'].class_to_idx
idx_to_class = { class_to_idx[cl] : cl for cl in class_to_idx }

#print(data_parts)
#print('train size:', dataset_sizes['train'])
#print('valid size:', dataset_sizes['valid'])
print('classes:', class_names)
print('class_to_idx:', class_to_idx)
print('idx_to_class:', idx_to_class)

#for i, (x, y) in enumerate(dataloaders['valid']):
#    print(x) # image
#    print(i, y) # image label



model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
model = model_ft

model_path = 'mymodel.pt'
model.load_state_dict(torch.load(model_path))
model.eval()


def inference(model, img_file, k=1):

    #IMAGE_SIZE = (224,224)    
    img = Image.open(img_file)
    #img = img.resize(IMAGE_SIZE)
    img = data_transforms['valid'](img)
    #inputs = Variable(img, volatile=True)
    inputs = Variable(img, requires_grad=False)
    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))
    
    inputs = inputs.to(device)
    outputs = model(inputs) # inference
    
    output = outputs[0].detach().cpu().numpy()
    #print(output)
    predict = np.argmax(output)
    topk_predicts = list(output.argsort()[::-1][:k])
    return predict, topk_predicts



if False:
    img_file = 'images/31.jpg'
    class_name = '31'
    predict, topk_predicts = inference(model, img_file, k=3)
    topk_class_names = list(map(lambda idx: idx_to_class[idx], topk_predicts))
    print('predict: {} (idx={})'.format(idx_to_class[predict], predict))
    print('top-6:', topk_predicts)
    print('topk_class_names:', topk_class_names)

#x = np.asarray(img)
#x = torch.tensor(x)
# then put it on the GPU, make it float and insert a fake batch dimension
#x = torch.Variable(x.cuda())
#x = x.float()
#x = x.unsqueeze(0)

"""
x = torch.randn(1, 3, 224, 224) 
y = model(x)
y = y[0]
print(y)
"""


# model testing
def model_testing(model, src_dir):

    src_dir = src_dir.rstrip('/')
    subdirs = os.listdir(src_dir)

    res1_list = []
    res6_list = []

    for class_name in subdirs:
        subdir = src_dir + '/' + class_name
        if not os.path.isdir(subdir): continue
        file_names = os.listdir(subdir)
        num_files = len(file_names)    
        print('\nclass={}, num_files={}'.format(class_name, num_files))

        for file_name in file_names:
            file_path = subdir + '/' + file_name

            predict, topk_predicts = inference(model, file_path, k=6)
            topk_predict_classes = list(map(lambda idx: idx_to_class[idx], topk_predicts))

            predict_class = idx_to_class[predict]

            res1 = 1 if predict_class == class_name else 0
            res6 = 1 if class_name in topk_predict_classes else 0
            res1_list.append(res1)
            res6_list.append(res6)

            print(file_path)
            print('[{}] -- class={}, predict={} (idx={})'.\
                format(res1, class_name, predict_class, predict))
            print('[{}] -- topk:{}'.format(res6, topk_predict_classes))
            #print('[{}] -- topk:{}  ({})'.\
            #    format(res6, topk_classes, topk_predicts))

    return np.mean(res1_list), np.mean(res6_list)

#test_dir = '/w/WORK/ineru/06_scales/_dataset/splited/test'
test_dir = '/home/andrei/Data/Datasets/Scales/splited/test'
top1, top6 = model_testing(model, test_dir)
print('\nRESULT: top1={:.4f}, top6={:.4f}'.format(top1, top6))