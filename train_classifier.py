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
#from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy

import progressbar
SHOW_BAR = False

from nn_models import get_resnet18_classifier, get_torchvision_model
from nn_models import get_small_cnn_classifier
from nn_models import CNN_Net

import data_factory
import settings
data_dir = settings.data_dir 


num_epochs = 5; start_lr = 0.002; step_size = 2


#root = '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018/'
#data_dir = '/w/WORK/ineru/06_scales/_dataset/splited/'

dataloaders, image_datasets = data_factory.load_data(data_dir)
#data_parts = list(dataloaders.keys())
dataset_sizes, class_names = data_factory.dataset_info(image_datasets)
num_classes = len(class_names)
data_parts = ['train', 'valid']

num_batch = dict()
num_batch['train'] = math.ceil(dataset_sizes['train'] / settings.batch_size)
num_batch['valid'] = math.ceil(dataset_sizes['valid'] / settings.batch_size)
print('train_num_batch:', num_batch['train'])
print('valid_num_batch:', num_batch['valid'])

#print(data_parts)
#print('train size:', dataset_sizes['train'])
#print('valid size:', dataset_sizes['valid'])
#print('classes:', class_names)
#print('class_to_idx:', dataset.class_to_idx)

#for i, (x, y) in enumerate(dataloaders['valid']):
#    print(x) # image
#    print(i, y) # image label


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    history = dict()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        history[epoch] = {}

        # Each epoch has a training and validation phase
        for phase in data_parts:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if SHOW_BAR: bar = progressbar.ProgressBar(maxval=num_batch[phase]).start()

            # Iterate over data.
            for i_batch, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if SHOW_BAR: bar.update(i_batch)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    #print('outputs: ', outputs)    

                # statistics
                if phase == 'valid':
                    print('preds: ', preds)
                    print('labels:', labels.data)
                    print('match: ', int(torch.sum(preds == labels.data)))

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if SHOW_BAR: bar.finish()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            history[epoch][phase] = {'loss': epoch_loss, 'acc': epoch_acc}
            if phase == 'valid':
                l_rate = scheduler.get_last_lr()
                print("l_rate: {}".format(l_rate))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    print("\nEp | TrLoss ValLoss | TrAcc ValAcc")
    for epoch in range(num_epochs):
        train_loss = history[epoch]['train']['loss']
        valid_loss = history[epoch]['valid']['loss']
        train_acc = history[epoch]['train']['acc']
        valid_acc = history[epoch]['valid']['acc']
        print('{}:   {:.4f} {:.4f} | {:.3f} {:.3f}'.format(epoch, train_loss, valid_loss, train_acc, valid_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    #model_ft = models.resnet18(pretrained=True)
    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, num_classes)

    #model = get_resnet18_classifier(num_classes)
    model = get_torchvision_model(num_classes)
    #model = CNN_Net(num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=0.5)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=num_epochs)

    # save model
    torch.save(model.state_dict(), "model_state.pt")
    #torch.save(model, "model_full.pt")
    torch.save(model, "model_full.pt", _use_new_zipfile_serialization=False)



"""
resnet-18:
Ep: TrainLoss ValLoss | TrainAcc ValAcc
0: 0.1703 0.0896 | 0.935 0.962
1: 0.0187 0.1034 | 0.996 0.962
2: 0.0136 0.0982 | 0.997 0.962
3: 0.0115 0.1498 | 0.997 0.952

mobilenet_v2:
Training complete in 17m 34s
Best val Acc: 0.952381
Ep: TrainLoss ValLoss | TrainAcc ValAcc
0: 0.2474 0.1890 | 0.905 0.952
1: 0.0290 0.1570 | 0.995 0.952
2: 0.0213 0.1999 | 0.995 0.952

--
resnet-18:
Training complete in 20m 15s
Best val Acc: 0.971429
Ep: TrainLoss ValLoss | TrainAcc ValAcc
0: 0.2893 0.1105 | 0.887 0.962
1: 0.0396 0.0940 | 0.994 0.962
2: 0.0335 0.1043 | 0.993 0.971

2) not pretrained:
Ep: TrainLoss ValLoss | TrainAcc ValAcc
0: 0.6930 0.5204 | 0.690 0.810
1: 0.2177 0.2041 | 0.934 0.933
2: 0.1376 0.3998 | 0.961 0.829 

3) resnet pretrained:
Ep: TrainLoss ValLoss | TrainAcc ValAcc
0: 0.3151 0.0990 | 0.882 0.952
1: 0.0672 0.0717 | 0.983 0.962
2: 0.0419 0.0820 | 0.987 0.971


"""