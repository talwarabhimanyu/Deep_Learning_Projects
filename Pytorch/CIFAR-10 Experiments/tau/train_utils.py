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
    def on_train_begin(self): pass
    def on_batch_begin(self): pass
    def on_batch_end(self): pass

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
    def __init__(self, optimizer, min_lr=1e-5, max_lr=10.0, num_iter=100):
        self.step = (log10(max_lr) - log10(min_lr))/num_iter
        self.optimizer = optimizer
        self.init_exp = log10(min_lr)
        self.curr_exp = self.init_exp
        self.smooth_loss = []
        self.lr_series = []
        self.iter_num = 0
        self.avg_loss = 0
        self.mom = 0.98
        self.curr_lr = 10**self.init_exp
        self.initLR()

    def initLR(self):
        for param in self.optimizer.param_groups:
            param['lr'] = self.curr_lr

    def updateLR(self, stats):
        self.iter_num += 1
        batch_loss = stats['train_loss']
        self.avg_loss = self.avg_loss*self.mom + batch_loss*(1 - self.mom)
        self.smooth_loss.append(self.avg_loss/(1 - self.mom**self.iter_num))
        self.lr_series.append(self.curr_lr)
        self.curr_exp += self.step
        self.curr_lr = 10**self.curr_exp
        for param in self.optimizer.param_groups:
            param['lr'] = self.curr_lr

    def on_batch_end(self, stats):
        self.updateLR(stats)

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
        self.plotLR(xlim=[min_lr, max_lr])

class LR_Circular(LR_Scheduler):
    def __init__(self):
        pass

class StatRecorder(Callback):
    def __init__(self, device, model, criterion, data_loaders, stat_list=['train_loss', 'val_loss'], val_freq_batches=5):
        self.iter_num = 0
        self.epoch_num = 0
        self.device = device
        self.model = model
        self.criterion = criterion
        self.data_loaders = data_loaders
        self.train_stat_list = list(filter(lambda st: 'train' in st, stat_list))
        self.train_stat_dict = {st : [] for st in self.train_stat_list}
        self.val_stat_list = list(filter(lambda st: 'val' in st, stat_list))
        self.val_stat_dict = {st : [] for st in self.val_stat_list}
        self.val_freq_batches = val_freq_batches
        self.val_iter_indices = []

    def updateIter(self):
        self.iter_num += 1

    def updateStats(self, stats):
        for st in self.train_stat_list:
            if (st == 'train_loss'):
                self.train_stat_dict[st].append(stats['train_loss'])
        if (self.iter_num % self.val_freq_batches == 0) and (len(self.val_stat_list) != 0):
            self.val_iter_indices.append(self.iter_num)
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
            

    def plotLossCurve(self):
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(8,5)
        fig.set_dpi(80)
        if 'val_loss' in self.val_stat_list:
            xticks = self.val_iter_indices
            train_loss_series = [self.train_stat_dict['train_loss'][i] for i in xticks]
        else:
            xticks = range(len(self.train_stat_dict['train_loss']))
            train_loss_series = self.train_stat_dict['train_loss']
        train_line = ax.plot(xticks, train_loss_series)
        if 'val_loss' in self.val_stat_list:
            val_line = ax.plot(xticks, self.val_stat_dict['val_loss'])

    def on_batch_end(self, stats):
        self.updateStats(stats)
        self.updateIter()


class NeuralNet():
    """

    Class Attributes:

    optimizer: initialized to SGD with lr=0.01 for all model parameters. This can
    be modified via loadOptim()

    Class Methods:

    loadOptim(optim_name, params): Loads an optimizer of type optim_name and initializes
    it using params.
        optim_name: String. One of 'sgd', 'adam'.
        params: It is of type parameters or a dictionary specifying parameter groups with or
        without group lrs.
    """
    def __init__(self, device, model_name, criterion_name, num_classes, data_loaders, pre_trained=False):
        self.device = device
        self.model_name = model_name
        self.criterion_name = criterion_name
        self.scheduler = None
        self.num_classes = num_classes
        self.data_loaders = data_loaders
        self.pre_trained = pre_trained
        self.model = self.loadModel()
        self.optimizer = self.loadOptim('default', self.model.parameters())
        self.criterion = self.loadCriterion()
        self.stat_recorder = StatRecorder(self.device, self.model, self.criterion, self.data_loaders)
        self.callbacks = [self.stat_recorder]

    def loadModel(self):
        """
        Loads model architecture along with pre-trained parameters (if asked). Replaces
        plain vanilla pooling with adaptive pooling for resnet models, and sets the 
        number of output nodes in the final layer equal to number of classes.

        """
        if (self.model_name == 'resnet18'):
            base_model = models.resnet18(pretrained=self.pre_trained)
        elif (self.model_name == 'resnet34'):
            base_model = models.resnet34(pretrained=self.pre_trained)
        elif (self.model_name == 'resnet50'):
            base_model = models.resnet50(pretrained=self.pre_trained)
        else:
            pass
        if ('resnet' in self.model_name):
            base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            base_model.fc = nn.Linear(base_model.fc.in_features, self.num_classes)
        return base_model

    def resetModel(self):
        self.model = self.loadModel()
    
    def getModel(self):
        return self.model

    def loadOptim(self, optim_name, optim_params, **kwargs):
        """
        Loads an optimizer for this model.

        """
        default_lr = 0.01
        if 'default_lr' in kwargs:
            default_lr = kwargs['default_lr']
        default_mom = 0.0
        if 'default_mom' in kwargs:
            default_mom = kwargs['default_mom']

        if (optim_name == 'default'):
            optimizer = optim.SGD(optim_params, lr=default_lr, \
                    momentum=default_mom)
        elif (optim_name == 'sgd'):
            optimizer = optim.SGD(optim_params, lr=default_lr, \
                    momentum=default_mom)
        elif (optim_name == 'adam'):
            optimizer = optim.Adam(optim_params, lr=default_lr)
        else:
            pass
        return optimizer

    def setOptim(self, optim_name, optim_params, **kwargs):
        self.optimizer = self.loadOptim(optim_name, optim_params, **kwargs)

    def getOptim(self):
        return self.optimizer

    def setScheduler(self, sched_name, **kwargs):
        if sched_name == 'finder':
            self.scheduler = LR_Finder(self.optimizer, **kwargs)
   
    def setCallbacks(self, callbacks):
        self.callbacks = callbacks

    def loadCriterion(self):
        if (self.criterion_name == 'cross_entropy'):
            criterion = nn.CrossEntropyLoss()
        else:
            pass
        return criterion
    
    def getCriterion(self):
        return self.criterion

    def freezeParams(self, freeze_modules=[]):
        """
        Freezes (sets requires_grad=FalsE) parameters of the model, based on arguments 
        provided:
        (1) freeze_modules: list of module names - all params belonging to these
            modules are frozen.

        """
        # First set requires_grad=True for all parameters.
        for param in self.model.parameters():
            param.requires_grad = True
        # Iterate over freeze_modules to set their requires_grad=False
        for name, module in self.model.named_modules():
            if (name in freeze_modules):
                for param in module.parameters():
                    param.requires_grad = False
        # Recreate optimizer with only those parameters for which requires_grad = True
        """ Add code to handle param_groups """
        self.optimizer = self.loadOptim(filter(lambda p: p.requires_grad, self.model.parameters()))

    def exploreLR(self, num_iter=200, min_lr=1e-4, max_lr=10.0):
        # Save current state of model and optimizer - they can be reverted to
        # original states if required.
        optim_state_dict = self.optimizer.state_dict
        model_state_dict = self.model.state_dict()
        self.resetModel()
        self.setOptim('sgd', self.model.parameters())
        self.setScheduler('finder', min_lr=min_lr, max_lr=max_lr, num_iter=num_iter)
        self.setCallbacks([self.scheduler])
        # Revert to original states here if required.

        # train here for num_iter
        trainer = Trainer(self.device, self, model_path='')
        trainer.train(self.data_loaders, num_iter=num_iter, iter_type='batch')

        return self.scheduler

    def printModules(self, verbose=True):
        # Display names of modules of this model instance. If verbose is True, then
        # only display immediate children modules, otherwise display a nested heirachy
        # of depth 4.
        for name, module in self.model.named_children():
            print('Module: ' + name)
            if not verbose:
                for name1, module1 in module.named_children():
                    print((8*' ') + '|_' + name1)
                    for name2, module2 in module1.named_children():
                        print((8*' ') + '|___' + name2)
                        for name3, module3 in module2.named_children():
                            print((8*' ') + '|_____' + name3)
                            for name4, module4 in module3.named_children():
                                print((8*' ') + '|_______ ' + name4)


# Given a model (an object inherited from nn.Module), this model will handle 
# training, hyperparameter tuning, evaluation, logging information.
class Trainer():
    """

    Class attributes:

    model_arch: The model architecture.
    """
    def __init__(self, device, model_arch, model_path):
        self.device = device
        self.model = model_arch.getModel()
        self.criterion = model_arch.getCriterion()
        self.optimizer = model_arch.getOptim()
        if model_arch.callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = model_arch.callbacks
    def train(self, data_loaders, num_iter=1, iter_type='epoch', 
            checkpoint_per_epoch=1, eval_stats=['train_loss', 'valid_loss'], stat_freq_batches=1, update_prog_bar_iter=5,
            verbose=True):
        """
        This method allows training of a model based on a given optimizer and criterion. The training
        can be run for a specified number of epochs (one epoch is full pass over training data) or for
        a specified number of iterations.
        Arguments:
        - iter_type: set to 'epoch' (trains for num_iter epochs) or 'batch' (runs for num_iter batches).
        - eval_stats: it can calculate the following statistics, every stat_freq_batches number of batches.
          'train_loss': Loss per batch averaged over stat_freq_batches number of batches.
          'valid_loss' : Loss per batch averaged over the entire validation set.
          'train_acc' : Accuracy for stat_freq_batches number of batches.
          'valid_acc' : Accuracy for the entire validation set.

        """
        progress_bar = tqdm(total=num_iter)
        iter_per_epoch = len(data_loaders)
        checkpoint_freq = floor(iter_per_epoch/checkpoint_per_epoch)
        #running_loss = 0.0
        torch.set_grad_enabled(True)
        iter_count = 0
        max_iters = 1e6
        max_epochs = 1e6
        stats = {}
        train_eval_stats = list(filter(lambda st: 'train' in st, eval_stats))
        valid_eval_stats = list(filter(lambda st: 'valid' in st, eval_stats))
        for st in eval_stats:
            stats.update({st : []})
        epoch_desc = 'e {}'
        if (iter_type == 'epoch'):
            max_epochs = num_iter
        else:
            max_iters = num_iter
            epoch_desc = ''
        epoch = 0

        for cb in self.callbacks: cb.on_train_begin()
        while epoch < max_epochs:
            description_str = epoch_desc.format(epoch + 1) + 'i {}'
            for iter_num, data in enumerate(data_loaders['train'], 1):
                if verbose and (iter_num%update_prog_bar_iter == 0):
                    progress_bar.set_description(description_str.format(iter_num))
                for cb in self.callbacks: cb.on_batch_begin()
                inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # backward pass
                loss.backward()
                self.optimizer.step()
                #running_loss += loss.item()
                stats = {'train_loss' : loss.item()}
                for cb in self.callbacks: cb.on_batch_end(stats)
                # eval stats, display stats, checkpoint if needed.
                #if (iter_num%stat_freq_batches == 0):
                    #valid set stats
                    #valid_stats = Trainer._evalStats(self.device, self.model, self.criterion, 
                    #        data_loaders['valid'], eval_stats=[x.replace('valid_', '') for x in valid_eval_stats])
                    #for st in valid_eval_stats:
                    #    stats[st].append(valid_stats[st.replace('valid_', '')])
                    #train_loss = running_loss/stat_freq_batches
                    ## train set stats
                    #if ('train_loss' in train_eval_stats):
                    #    stats['train_loss'].append(train_loss)
                    #running_loss = 0.0
                    #status_dict = {'train_loss' : '{:.2f}'.format(train_loss)}
                    #if ('loss' in valid_stats):
                    #    status_dict.update({'val_loss' : ':.2f'.format(valid_stats['loss'])})
                    #progress_bar.set_postfix(status_dict) 
                if (iter_num%checkpoint_freq == 0):
                    #checkpoint
                    pass
                iter_count += 1
                if (iter_count == max_iters):
                    break
            if verbose: progress_bar.update(1)
            if (iter_count == max_iters):
                break
            epoch += 1
       # return stats

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
