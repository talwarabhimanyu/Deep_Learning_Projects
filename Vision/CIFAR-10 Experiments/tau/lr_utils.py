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
import imageio

class Callback():
    def on_train_begin(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass

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

    def iter_time(self):
        return self.iter_num
    
    def epoch_time(self):
        return self.epoch_num


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

    def plot_findings(self, xlim=None):
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
        self.plot_findings(xlim=[min_lr, max_lr])

class LR_Cyclical(LR_Scheduler):
    """
    This implements Leslie Smith's Cyclical LR regim. The original paper can be found here:
    https://arxiv.org/abs/1506.01186
    Class Attributes:
    (1) cycle_len: number of iterations after which the learning rate reverts to min_lr
    (2) policy: one of 'triangle' or 'exp', it describes how learning rate varies over one cycle
    (3) min_lr/max_lr: specify the range over which the learning rate varies over a cycle

    """
    def __init__(self, optimizer, clock, cycle_len, cycle_mult=1, min_lr=1e-5, max_lr=1.0, policy='triangle'):
        self.clock = clock
        self.optimizer = optimizer
        self.cycle_mult = cycle_mult
        # Add 1 to cycle len to make it an odd number
        self.cycle_len = cycle_len + (cycle_len+1)%2
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.policy = policy
        self.lr_series = []
        self.curr_lr = max_lr
        if policy == 'triangle': self.step_size = (max_lr - min_lr)*2/(self.cycle_len - 1)
        # The first update of a cycle receives a counter of 0
        self.counter = 0
    
    def updateLR(self, **kwargs):
        if self.policy == 'triangle':
            if self.counter == 0:                   self.curr_lr = self.min_lr
            elif self.counter < 0.5*self.cycle_len: self.curr_lr += self.step_size
            else:                                   self.curr_lr -= self.step_size
        elif self.policy == 'cosine':
            self.curr_lr = self.min_lr + (self.max_lr - self.min_lr)*(1 + np.cos((self.counter % self.cycle_len*np.pi/(self.cycle_len-1))))/2
        else:
            raise NotImplementedError
        for param in self.optimizer.param_groups:
            param['lr'] = self.curr_lr
        self.lr_series.append(self.curr_lr)
        self.counter += 1
        if (self.counter % self.cycle_len == 0) and (self.clock.iter_time() > 1): self.resetCycle()

    def resetCycle(self):
        self.cycle_len = self.cycle_len*self.cycle_mult
        self.cycle_len += (self.cycle_len + 1)%2
        if self.policy == 'triangle':
            self.step_size = (self.max_lr - self.min_lr)*2/(self.cycle_len - 1)
        self.counter = 1
    
    def on_batch_begin(self, **kwargs):
        self.updateLR(**kwargs)
    
class WeightWatcher(Callback):
    def __init__(self, device, clock, optimizer):
        self.optimizer = optimizer
        self.device = device
        self.clock = clock
        self.epoch_count = 0
    
    def saveWeights(self):
        layer_matrix = self.optimizer.param_groups[0]['params'][0].detach().cpu().numpy()
        num_filters = layer_matrix.shape[0]
        layer_matrix = layer_matrix.transpose(0,2,3,1)
        num_cols = 8
        num_rows = int(num_filters/num_cols)
        plt.ioff()
        fig, ax = plt.subplots(num_rows, num_cols)
        fig.set_size_inches(num_cols*1.25, num_rows*1.25)
        for i in range(num_filters):
            im = layer_matrix[i, :, :, :]
            im = im - im.min()
            im = im/im.max()
            r = i // num_cols
            c = i % num_cols
            ax[r,c].imshow(im)
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
        self.epoch_count += 1
        plt.suptitle('Epoch {}'.format(self.epoch_count), fontsize=18)
        plt.savefig('epoch_{}_weights.png'.format(self.clock.epoch_time()), pad_inches=0.02)
        plt.close(fig)
        plt.ion()
    
    def makeGIF(self):
        images = []
        for i in range(self.epoch_count):
            images.append(imageio.imread('epoch_{}_weights.png'.format(i+1)))
        imageio.mimsave('weights.gif', images, duration=0.2)

    def on_epoch_end(self):
        self.saveWeights()
    
    def on_train_end(self, **kwargs):
        self.makeGIF()

class StatRecorder(Callback):
    def __init__(self, device, clock, model, criterion, data_loaders, save_stats, stat_list=['train_loss', 'val_loss'], val_freq_per_epoch=10):
        self.device = device
        self.model = model
        self.clock = clock
        self.save_stats = save_stats
        self.criterion = criterion
        self.data_loaders = data_loaders
        self.train_stat_list = list(filter(lambda st: 'train' in st, stat_list))
        self.train_stat_dict = {st : [] for st in self.train_stat_list}
        self.val_stat_list = list(filter(lambda st: 'val' in st, stat_list))
        self.val_stat_dict = {st : [] for st in self.val_stat_list}
        batch_size = data_loaders['train'].batch_size
        epoch_size_in_batches = len(data_loaders['train'].dataset)/batch_size
        self.val_freq_batches = int(epoch_size_in_batches/val_freq_per_epoch)
        if self.val_freq_batches == 0: self.val_freq_batches = 1
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
                self.running_corrects += torch.sum(preds == stats_dict['labels'].data).item()
        if (self.clock.iter_num % self.val_freq_batches == 0):
            # Do a complete pass over the validation set to calculate validation stats
            if (len(self.val_stat_list) != 0):
                self.val_iter_indices.append(self.clock.iter_num)
                torch.set_grad_enabled(False)
                val_iter = 0
                cum_val_loss = 0
                cum_val_count = 0
                cum_val_corrects = 0
                for data in iter(self.data_loaders['val']):
                    inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                    outputs = self.model(inputs)
                    if ('val_loss' in self.val_stat_list):
                        loss = self.criterion(outputs, labels)
                        cum_val_loss += loss.item()
                    if 'val_acc' in self.val_stat_list:
                        _, preds = torch.max(outputs, 1)
                        cum_val_count += preds.size(0)
                        cum_val_corrects += torch.sum(preds == labels.data).item()
                    val_iter += 1
                torch.set_grad_enabled(True)
                if ('val_loss' in self.val_stat_list):
                    self.val_stat_dict['val_loss'].append(cum_val_loss/val_iter)
                if ('val_acc' in self.val_stat_list):
                    self.val_stat_dict['val_acc'].append(cum_val_corrects/cum_val_count)
            # Calculate running train stats
            if 'train_acc' in self.train_stat_list:
                self.train_stat_dict['train_acc'].append(self.running_corrects / self.running_count)
                self.running_count = 0
                self.running_corrects = 0

    def getLatestStats(self):
        last_iter = 0
        if len(self.val_iter_indices) != 0:
            last_iter = self.val_iter_indices[-1]
        latest_stats = {'train_loss' : self.train_stat_dict['train_loss'][last_iter - 1]}
        if 'val_loss' in self.val_stat_dict:
            latest_stats.update({'val_loss' : self.val_stat_dict['val_loss'][-1]})
        if 'val_acc' in self.val_stat_dict:
            latest_stats.update({'val_acc' : self.val_stat_dict['val_acc'][-1]})
        if 'train_acc' in self.train_stat_dict:
            latest_stats.update({'train_acc' : self.train_stat_dict['train_acc'][-1]})
        return latest_stats

    def plotLoss(self):
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(8,5)
        fig.set_dpi(80)
        train_line = ax.plot(range(len(self.train_stat_dict['train_loss'])), \
                self.train_stat_dict['train_loss'], linewidth=1.0, color='red', \
                label='train_loss')
        if 'val_loss' in self.val_stat_list:
            val_line = ax.plot(np.asarray(self.val_iter_indices) - 1, self.val_stat_dict['val_loss'], \
                    linewidth=1.25, color='green', label='val_loss')
        _ = ax.legend(loc='upper right')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss per Batch')

    def plotAccuracy(self):
        if 'train_acc' in self.train_stat_dict and 'val_acc' in self.val_stat_dict:
            fig = plt.gcf()
            ax = plt.gca()
            fig.set_size_inches(8,5)
            fig.set_dpi(80)
            train_line = ax.plot(range(len(self.train_stat_dict['train_acc'])), \
                    self.train_stat_dict['train_acc'], linewidth=1.0, color='red', \
                    label='train_acc')
            val_line = ax.plot(np.asarray(self.val_iter_indices) - 1, self.val_stat_dict['val_acc'], \
                    linewidth=1.25, color='green', label='val_acc')
            _ = ax.legend(loc='best')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')


    def writeStats(self, **kwargs):
        if self.save_stats:
            config_name = kwargs['config_name']
            with open(config_name + '_train_loss.csv', 'w') as f:
                writer = csv.writer(f,delimiter=",", lineterminator='\n')
                writer.writerow(['iteration','train_loss'])
                for idx, loss in enumerate(self.train_stat_dict['train_loss']):
                    row = ['{}'.format(idx+1),'{:.4f}'.format(loss)]
                    writer.writerow(row)
            if self.val_stat_dict:
                self.val_stat_dict.update({'iteration':self.val_iter_indices})
                if 'train_acc' in self.train_stat_dict:
                    self.val_stat_dict.update({'train_acc':self.train_stat_dict['train_acc']})

                with open(config_name + '_other_stats.csv', 'w') as f:
                    writer = csv.writer(f, delimiter=",", lineterminator='\n')
                    writer.writerow(self.val_stat_dict.keys())
                    writer.writerows(zip(*self.val_stat_dict.values()))

    def on_train_begin(self, **kwargs):
        self.resetRecorder()
   
    def on_batch_end(self, **kwargs):
        self.updateStats(**kwargs)

    def on_train_end(self, **kwargs):
        # Write stats to file
        self.writeStats(**kwargs)

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
