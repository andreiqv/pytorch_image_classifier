#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First run split_dataset.py to split dataset in train|valid|test parts.
Then run this script.

source /mnt/ext1/venv/torch/bin/activate
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
from scipy.special import softmax

from nn_models import get_resnet18_classifier
from nn_models import get_small_cnn_classifier
from nn_models import CNN_Net

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


"""
#model_ft = models.resnet18(pretrained=True)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, num_classes)
model = get_resnet18_classifier(num_classes)
#model = CNN_Net(num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model_path = "model_state.pt"
model.load_state_dict(torch.load(model_path))
model.eval()
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
full_model = torch.load("model_full.pt")
full_model = full_model.to(device)
print("Loading model is done.")


def inference(model, img, k=1):

    #IMAGE_SIZE = (224,224)
    #img = img.resize(IMAGE_SIZE)
    #print("img size:", img.size)

    img = data_transforms['valid'](img)
    #inputs = Variable(img, volatile=True)
    inputs = Variable(img, requires_grad=False)
    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))
    
    inputs = inputs.to(device)
    outputs = model(inputs) # inference
    
    output = outputs[0].detach().cpu().numpy()
    # print(output)
    normalized_output = softmax(output)
    # print("normalized_output:", normalized_output)
    max_index = np.argmax(output)
    max_prob = normalized_output[max_index]

    topk_predicts = list(output.argsort()[::-1][:k])
    
    return max_index, topk_predicts, max_prob



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

            predict, topk_predicts, max_prob = inference(model, file_path, k=6)
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



def inference_directory(in_dir, model):
    """ in_dir - a directory with pictures
    """
    with open("_inference_output.txt", "wt") as outfp:

        for filename in os.listdir(in_dir):
            #print("\n{}".format(filename))
            path = os.path.join(in_dir, filename)
            img = Image.open(path)
            rgb_img = Image.new("RGB", img.size)
            rgb_img.paste(img)
            rgb_img = rgb_img.resize((224, 224))
            t1 = time.time()
            predict, topk_predicts, max_prob = inference(model, rgb_img)
            t2 = time.time()
            topk_class_names = list(map(lambda idx: idx_to_class[idx], topk_predicts))
            class_name = idx_to_class[predict]
            #print("Inference time = {:.2f}".format(t2 - t1))
            #print('predict: {} (idx={})'.format(idx_to_class[predict], predict))
            #print("max_prob: {:.2f}".format(max_prob))
            #print('top-6:', topk_predicts)
            #print('topk_class_names:', topk_class_names)
            print('class={} (idx={}), prob={:.2f}, in {:.2f} sec. - {}'.format(class_name, predict, max_prob, t2 - t1, filename))
            outfp.write("{} [{:.2f}] - {}\n".format(class_name, max_prob, filename))



if __name__ == "__main__":

    print(sys.argv)

    if len(sys.argv) > 1:
        in_dir = sys.argv[1]
    else:
        #in_dir = "../test2/"
        #in_dir="../dataset_abs/valid/1/"
        in_dir = "/data/5_patexia_2023/41_classiffier/dataset/train/1/"

    inference_directory(in_dir=in_dir, model=full_model)
    
    """
    img_file = '/data/5_patexia/image_classifier/0190_TRNA.png'
    #img_file = '/data/5_patexia/image_classifier/INTV.png'
    #class_name = '31'
    img = Image.open(img_file)
    rgb_img = Image.new("RGB", img.size)
    rgb_img.paste(img)
    rgb_img = rgb_img.resize((224, 224))
    #rgb_img.show()

    t1 = time.time()
    predict, topk_predicts, max_prob = inference(model, rgb_img)
    t2 = time.time()
    print("Inference time = {:.2f}".format(t2 - t1))

    topk_class_names = list(map(lambda idx: idx_to_class[idx], topk_predicts))
    print('predict: {} (idx={})'.format(idx_to_class[predict], predict))
    print("max_prob: {:.2f}".format(max_prob))
    print('top-6:', topk_predicts)
    print('topk_class_names:', topk_class_names)

    #x = np.asarray(img)
    #x = torch.tensor(x)
    # then put it on the GPU, make it float and insert a fake batch dimension
    #x = torch.Variable(x.cuda())
    #x = x.float()
    #x = x.unsqueeze(0)
    """

    """
    x = torch.randn(1, 3, 224, 224) 
    y = model(x)
    y = y[0]
    print(y)
    """

    """
    #test_dir = '/w/WORK/ineru/06_scales/_dataset/splited/test'
    test_dir = '/home/andrei/Data/Datasets/Scales/splited/test'
    top1, top6 = model_testing(model, test_dir)
    print('\nRESULT: top1={:.4f}, top6={:.4f}'.format(top1, top6))
    """
