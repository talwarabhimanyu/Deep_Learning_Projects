import numpy as np
from math import floor, log10
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

class Callback():
    def on_train_begin(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass

class Clock(Callback):
    """
    Universal clock for a model, shared across all attributes and methods
    of a model instance. Clock increments by 1 at the BEGINNING of an epoch,
    or batch.
    """
    def __init_(self):
        self.iter_num = 0
        self.batch_num = 0
        self.epoch_num = 0
    
    def resetClock(self):
        self.iter_num = 0
        self.batch_num = 0
        self.epoch_num = 0
    
    def on_train_begin(self, **kwargs):
        self.resetClock()

    def on_epoch_begin(self, **kwargs):
        self.epoch_num += 1
        self.batch_num = 0

    def on_batch_begin(self, **kwargs):
        self.iter_num += 1
        self.batch_num += 1


class LR_Scheduler(Callback):
    """
    Abstract class which will be inherited by every LR Scheduler class
    and the LR Finder class. I have borrowed ideas related to Callbacks
    from Fast.AI's library. The code files I've consulted can be found
    here:
    https://github.com/fastai/fastai/blob/master/fastai/model.py
    https://github.com/fastai/fastai/blob/master/fastai/sgdr.py
    https://github.com/fastai/fastai/blob/master/fastai/learner.py

    """
    def updateLR(self): pass

class LR_Finder(LR_Scheduler):
    def __init__(self, optimizer, clock, min_lr=1e-5, max_lr=10.0, num_iters=100):
        self.clock = clock
        self.step = (log10(max_lr) - log10(min_lr))/num_iters
        self.optimizer = optimizer
        self.init_exp = log10(min_lr)
        self.curr_exp = self.init_exp
        self.smooth_loss = []
        self.lr_series = []
        self.avg_loss = 0
        self.mom = 0.98
        self.curr_lr = 10**self.init_exp
        self.initLR()

    def initLR(self):
        for param in self.optimizer.param_groups:
            param['lr'] = self.curr_lr

    def updateLR(self, stats_dict):
        batch_loss = stats_dict['train_loss']
        self.avg_loss = self.avg_loss*self.mom + batch_loss*(1 - self.mom)
        self.smooth_loss.append(self.avg_loss/(1 - self.mom**self.clock.iter_num))
        self.lr_series.append(self.curr_lr)
        self.curr_exp += self.step
        self.curr_lr = 10**self.curr_exp
        for param in self.optimizer.param_groups:
            param['lr'] = self.curr_lr

    def on_batch_end(self, **kwargs):
        self.updateLR(**kwargs)

    def plotLR(self, xlim=None):
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(8,5)
        fig.set_dpi(80)
        _ = plt.plot(np.asarray(self.lr_series), np.asarray(self.smooth_loss))
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rates (log-scale)')
        ax.set_ylabel('Loss (Smoothed)')
        if xlim is not None:
            ax.set_xlim(left=xlim[0], right=xlim[1])
            max_idx = min(range(len(self.lr_series)), key=lambda i: abs(self.lr_series[i] - xlim[1]))
            min_idx = min(range(len(self.lr_series)), key=lambda i: abs(self.lr_series[i] - xlim[0]))
            ax.set_ylim(top=np.asarray(self.smooth_loss[min_idx:max_idx]).max(), \
                    bottom=np.asarray(self.smooth_loss[min_idx:max_idx]).min())

    def zoomPlot(self, min_lr=1e-3, max_lr=1e-1):
        """
        This function zooms into the original LRs vs Smoothed Loss plot.
        """
        self.plotLR(xlim=[min_lr, max_lr])

class LR_Cyclical(LR_Scheduler):
    """
    This implements Leslie Smith's Cyclical LR regim. The original paper can be found here:
    https://arxiv.org/abs/1506.01186    

    """
    def __init__(self):
        pass

class StatRecorder(Callback):
    def __init__(self, device, clock, model, criterion, data_loaders, stat_list=['train_loss', 'val_loss'], val_freq_batches=5):
        self.device = device
        self.model = model
        self.clock = clock
        self.criterion = criterion
        self.data_loaders = data_loaders
        self.train_stat_list = list(filter(lambda st: 'train' in st, stat_list))
        self.train_stat_dict = {st : [] for st in self.train_stat_list}
        self.val_stat_list = list(filter(lambda st: 'val' in st, stat_list))
        self.val_stat_dict = {st : [] for st in self.val_stat_list}
        self.val_freq_batches = val_freq_batches
        self.val_iter_indices = []
        self.running_corrects = 0
        self.running_count = 0

    def setStatList(self, stat_list):
        if 'train_loss' not in stat_list:
            stat_list.append('train_loss')
        self.train_stat_list = list(filter(lambda st: 'train' in st, stat_list))
        self.val_stat_list = list(filter(lambda st: 'val' in st, stat_list))

    def resetRecorder(self):
        self.train_stat_dict = {st : [] for st in self.train_stat_list}
        self.val_stat_dict = {st : [] for st in self.val_stat_list}
        self.val_iter_indices = []
    
    def updateStats(self, stats_dict):
        for st in self.train_stat_list:
            if (st == 'train_loss'):
                self.train_stat_dict[st].append(stats_dict['train_loss'])
            elif (st == 'train_acc'):
                _, preds = torch.max(stats_dict['outputs'], 1)
                self.running_count += preds.size(0)
                self.running_corrects += torch.sum(preds == stats_dict['labels'].data)
        if (self.clock.iter_num % self.val_freq_batches == 0):
            # Do a complete pass over the validation set to calculate validation stats
            if (len(self.val_stat_list) != 0):
                self.val_iter_indices.append(self.clock.iter_num)
                torch.set_grad_enabled(False)
                val_iter = 0
                cum_val_loss = 0
                for data in iter(self.data_loaders['val']):
                    inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                    outputs = self.model(inputs)
                    if ('val_loss' in self.val_stat_list):
                        loss = self.criterion(outputs, labels)
                        cum_val_loss += loss.item()
                    val_iter += 1
                torch.set_grad_enabled(True)
                if ('val_loss' in self.val_stat_list):
                    self.val_stat_dict['val_loss'].append(cum_val_loss/val_iter)
            # Calculate running train stats
            if 'train_acc' in self.train_stat_list:
                self.train_stat_dict['train_acc'].append(self.running_corrects.double() / self.running_count)
                self.running_count = 0
                self.running_corrects = 0

    def getLatestStats(self):
        last_iter = -1
        if len(self.val_iter_indices) != 0:
            last_iter = self.val_iter_indices[-1]
        latest_stats = {'train_loss' : self.train_stat_dict['train_loss'][last_iter - 1]}
        if 'val_loss' in self.val_stat_dict:
            latest_stats.update({'val_loss' : self.val_stat_dict['val_loss'][-1]})
        if 'train_acc' in self.train_stat_dict:
            latest_stats.update({'train_acc' : self.train_stat_dict['train_acc'][-1]})
        return latest_stats

    def plotLoss(self):
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(8,5)
        fig.set_dpi(80)
        train_line = ax.plot(range(len(self.train_stat_dict['train_loss'])), \
                self.train_stat_dict['train_loss'], linewidth=1.25, color='red', \
                label='train_loss')
        if 'val_loss' in self.val_stat_list:
            val_line = ax.plot(np.asarray(self.val_iter_indices) - 1, self.val_stat_dict['val_loss'], \
                    linewidth=1.25, color='green', label='val_loss')
        _ = ax.legend(loc='upper right')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss per Batch')

    def on_train_begin(self, **kwargs):
        self.resetRecorder()
   
    def on_batch_end(self, **kwargs):
        self.updateStats(**kwargs)

class NotebookDisplay(Callback):
    def __init__(self, model_setup):
        self.stat_recorder = model_setup.stat_recorder
        self.clock = model_setup.clock
        self.stat_show_freq = self.stat_recorder.val_freq_batches
        self.every_epoch = True
        self.num_iters = 0
        self.prog_bar = None
        self.str_format = {'train_loss' : '{:.2f}',
                'val_loss' : '{:.2f}',
                'train_acc' : '{0:.2%}',
                'val_acc' : '{0:.2%}'}

    def initializeBar(self, num_iters, iter_type):
        if (iter_type == 'epoch'):
            self.every_epoch = True
        elif (iter_type == 'batch'):
            self.every_epoch = False
        else:
            self.every_epoch = True
        self.num_iters = num_iters
        self.prog_bar = tqdm(total=self.num_iters)
    
    def updateBarCount(self):
        self.prog_bar.update(1)

    def updateBarStats(self):
        stat_dict = self.stat_recorder.getLatestStats()
        show_dict = {}
        if self.every_epoch:
            show_dict.update({'batch' : '{}'.format(self.clock.batch_num)})
        show_dict.update({key : self.str_format[key].format(stat_dict[key]) \
                for key in stat_dict})
        self.prog_bar.set_postfix(show_dict)
    
    def on_batch_end(self, **kwargs):
        if not self.every_epoch:
            self.updateBarCount()
        if (self.clock.batch_num % self.stat_show_freq == 0):
            self.updateBarStats()

    def on_train_begin(self, **kwargs):
        self.initializeBar(**kwargs)

    def on_epoch_end(self):
        if self.every_epoch:
            self.updateBarCount()
