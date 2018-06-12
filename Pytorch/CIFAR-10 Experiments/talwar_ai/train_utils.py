import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import csv
import os
import shutil
import re
from tqdm import tqdm_notebook as tqdm
import gc


# Given a model (an object inherited from nn.Module), this model will handle 
# training, hyperparameter tuning, evaluation, logging information.
class ModelTrainer():
    def __init__(self, device, model, criterion, optimizer, model_path):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
    def Train(self, data_loaders, num_epochs=1, checkpoint_per_epoch=1, 
              eval_stats=['loss'], stat_per_epoch=1):
        progress_bar = tqdm(total=num_epochs)
        iter_per_epoch = len(data_loaders)
        stat_freq = floor(iter_per_epoch/stat_per_epoch)
        checkpoint_freq = floor(iter_per_epoch/checkpoint_per_epoch)
        running_loss = 0.0
        torch.set_grad_enabled(True)
        for epoch in range(num_epochs):
            for i, data in enumerate(data_loaders['train'], 1):
                if (i%5 == 0):
                    progress_bar.set_description('e {} i {}'.format(epoch + 1, i))
                inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # backward pass
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                # eval stats, display stats, checkpoint if needed.
                if (i%stat_freq == 0):
                    #eval stats
                    valid_stats = self._evalStats(self.device, self.model, self.criterion, 
                            data_loaders['valid'], eval_stats=eval_stats)
                    train_loss = running_loss/stat_freq
                    running_loss = 0.0
                    progress_bar.set_postfix(train_loss='{:.2f}'.format(train_loss), 
                                         val_loss='{:.2f}'.format(valid_stats['loss']))
                if (i%checkpoint_freq == 0):
                    #checkpoint
                    pass
            progress_bar.update(1)
    @classmethod
    def _evalStats(cls, device, model, criterion, data_loader, num_items=1, eval_stats=['loss']):
        num_iter = 0
        loss_stat= 0.0
        stats = {}
        torch.set_grad_enabled(False)
        for data in iter(data_loader):
            inputs, labels = data['image'].to(device), data['label'].long().to(device)
            outputs = model(inputs)
            if ('loss' in eval_stats):
                loss = criterion(outputs, labels)
                loss_stat += loss.item()
            num_iter += 1
        torch.set_grad_enabled(True)
        if ('loss' in eval_stats):
            if (num_iter == 0):
                loss_stat = 0.0
            else:
                loss_stat = round(loss_stat/num_iter*num_items, 2)
            stats.update({'loss' : loss_stat})
        return stats
        
    def SaveCheckpoint(self, model_path):
        state = {'model_dict' : self.model.state_dict(),
                'optim_dict' : self.optimizer.state_dict(),
                'state_epoch' : state_epoch}
        torch.save(state, model_path)
    def LoadCheckPoint(model_path):
        model_dict, optim_dict = None, None
        if os.path.isfile(model_path):
            if torch.cuda.is_available():
                state = torch.load(model_path)  
            else:
                state = torch.load(model_path,
                                       map_location=lambda storage,
                                       loc:storage)
            model_dict = state['model_dict']
            optim_dict = state['optim_dict']
            state_epoch = state['state_epoch']
        return model_dict, optim_dict, state_epoch

def LRPlots(num_trial=5, iter_per_trial=10):
    torch.set_grad_enabled(True)
    update_freq = 1
    plots = {}
    for trial in range(num_trial):
        lr = 10**(-np.random.uniform(1, 3))
        model, optimizer, criterion = GetModel(lr)
        running_loss = 0.0
        num_updates  = 0
        loss_record = []
        for i, data in zip(range(iter_per_trial), image_loaders['train']):
            inputs, labels = data['image'].to(device), data['label'].long().to(device)
            # forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            # update statistics
            running_loss += loss.item()
            num_updates += 1
            if (i%update_freq == 0):
                loss_record += [round(running_loss/num_updates, 2)]
                running_loss = 0.0
                num_updates = 0
        plots.update({round(lr, 5) : loss_record})
    return plots

def TestFn():
    print('Hello World')
