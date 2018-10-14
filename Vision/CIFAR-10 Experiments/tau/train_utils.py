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
from tau.lr_utils import *

class NeuralNet():
    """

    CLASS ATTRIBUTES:
    ==================

    pred_fn_class:  Type: String
                    Decsription: Name of the Prediction Function Class, such as 'resnet18', which
                    will be included in this model.

    pred_fn:        Type: nn.module
                    Description: the Prediction Function Class to be used. This will be an
                    instance of of pre-trained Pred. Functions such as torchvision.models.resnet18, 
                    or a custom Neural Net object which inherits from nn.module.

    optimizer:      Type: torch.optim.optimizer
                    Description: by default, initialized to SGD with lr=0.01 for all model 
                    parameters. Optimizer can be modified via self.setOptim()

    criterion:      Type: one of PyTorch's loss objects such as nn.CrossEntropyLoss

    CLASS METHODS:
    ================
    (See individual methods for docs on input and return types).

    loadOptim(optim_name, params): Loads an optimizer of type optim_name and initializes
    it using params.

        callbacks: This will be initialized with the Neural Net's clock and its stat_recorder. 
        These two callback must never be removed from the callbacks list. Additional
        callbacks can be added to the list using addCallbacks() method.

    """
    def __init__(self, device, pred_fn_class, criterion_name, num_classes, data_loaders, pre_trained=False, save_stats=True):
        self.device = device
        self.pred_fn_class = pred_fn_class
        self.criterion_name = criterion_name
        self.scheduler = None
        self.num_classes = num_classes
        self.data_loaders = data_loaders
        self.pre_trained = pre_trained
        self.pred_fn = self.loadPredFn()
        self.optimizer = self.loadOptim('sgd', self.pred_fn.parameters())
        self.criterion = self.loadCriterion()
        self.clock = Clock()
        self.stat_recorder = StatRecorder(self.device, self.clock, self.pred_fn, self.criterion, self.data_loaders, save_stats)
        self.callbacks = [self.clock, self.stat_recorder]
        self.batch_size = data_loaders['train'].batch_size
        self.epoch_size_in_batches = int(len(data_loaders['train'].dataset)/self.batch_size)


    def loadPredFn(self):
        """
        Loads model architecture along with pre-trained parameters (if asked). Replaces
        plain vanilla pooling with adaptive pooling for resnet models, and sets the 
        number of output nodes in the final layer equal to number of classes.

        """
        if (self.pred_fn_class == 'resnet18'):
            pred_fn = models.resnet18(pretrained=self.pre_trained)
        elif (self.pred_fn_class == 'resnet34'):
            pred_fn = models.resnet34(pretrained=self.pre_trained)
        elif (self.pred_fn_class == 'resnet50'):
            pred_fn = models.resnet50(pretrained=self.pre_trained)
        else:
            raise NotImplementedError
        if ('resnet' in self.pred_fn_class):
            pred_fn.avgpool = nn.AdaptiveAvgPool2d(1)
            pred_fn.fc = nn.Linear(pred_fn.fc.in_features, self.num_classes)
        pred_fn = pred_fn.to(self.device)
        return pred_fn

    def resetModel(self):
        self.pred_fn = self.loadPredFn()
        self.clock.resetClock()
    
    def getPredFn(self):
        return self.pred_fn

    def loadOptim(self, optim_name, optim_params, **kwargs):
        """
        Loads an optimizer for this intance into self.optimizer
        INPUT
        =======
        optim_name:     String, one of the following
                        * 'sgd'
                        * 'adam'

        optim_params:   torch.nn.parameter object
        
        **kwargs:       Keyword arguments
                        * 'lr':  float, initial learning 
                                 rate for sgd, adam
                                 Default: 0.01
                        * 'mom': float, momentum for
                                 sgd
                                 Default: 0.5

        RETURNS
        =======
        optimizer:      torch.optim.optimizer object

        """
        lr = 0.01
        if 'lr' in kwargs:
            lr = kwargs['lr']
        mom = 0.5
        if 'mom' in kwargs:
            mom = kwargs['mom']
        weight_decay = 0.0
        if 'weight_decay' in kwargs:
            weight_decay = kwargs['weight_decay']
        self.optim_name = 'optim({}, lr={}, mom={})'.format(optim_name, lr, mom)
        if (optim_name == 'sgd'):
            optimizer = optim.SGD(optim_params, lr=lr, \
                    momentum=mom, weight_decay=weight_decay)
        elif (optim_name == 'adam'):
            optimizer = optim.Adam(optim_params, lr=lr)
        else:
            raise NotImplementedError
        return optimizer

    def setOptim(self, optim_name, optim_params=None, **kwargs):
        if optim_params is None:
            optim_params = self.pred_fn.parameters()
        self.optimizer = self.loadOptim(optim_name, optim_params, **kwargs)

    def getOptim(self):
        return self.optimizer

    def setScheduler(self, sched_name, **kwargs):
        """
        INPUTS
        =======
        sched_name:     String, one of the following
                        * 'finder'
                        * 'cyclical', varies lr between [min_lr, max_lr]
                           which are specified as keyword args.
        **kwargs:       Key word arguments
                        * 'min_lr' float, used for 'cyclical'
                        * 'max_lr' float, used for 'cyclical'
                        * 'cycle_len' positive integer, used for 'cyclical'

        RETURNS
        =======
        Does not return anything. Sets self.scheduler for this instance to
        the appropriate scheduler object.

        """
        if sched_name == 'finder':
            self.scheduler = LR_Finder(self.optimizer, self.clock, **kwargs)
        elif sched_name == 'cyclical':
            if 'cycle_len' not in kwargs:
                cycle_len = self.epoch_size_in_batches*2
                kwargs.update({'cycle_len':cycle_len})
            self.scheduler = LR_Cyclical(self.optimizer, self.clock, **kwargs)
        self.setCallbacks(self.scheduler)
   
    def setCallbacks(self, cb):
        self.callbacks = [self.clock, self.stat_recorder]
        if isinstance(cb, list):
            for c in cb:
                if not c in self.callbacks: self.callbacks.append(cb)
        else:
            if not cb in self.callbacks: self.callbacks.append(cb)
    
    def setStatList(self, stat_list):
        self.stat_recorder.setStatList(stat_list)

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
        for param in self.pred_fn.parameters():
            param.requires_grad = True
        # Iterate over freeze_modules to set their requires_grad=False
        for name, module in self.pred_fn.named_modules():
            if (name in freeze_modules):
                for param in module.parameters():
                    param.requires_grad = False
        # Recreate optimizer with only those parameters for which requires_grad = True
        """ Add code to handle param_groups """
        self.optimizer = self.loadOptim(filter(lambda p: p.requires_grad, self.pred_fn.parameters()))

    def exploreLR(self, num_iters=200, min_lr=1e-4, max_lr=10.0):
        # Save current state of model and optimizer - they can be reverted to
        # original states if required.
        orig_optim_state_dict = self.optimizer.state_dict
        orig_model_state_dict = self.pred_fn.state_dict()
        orig_cbs = self.callbacks
        self.resetModel()
        self.setOptim('sgd', self.pred_fn.parameters())
        self.setScheduler('finder', min_lr=min_lr, max_lr=max_lr, num_iters=num_iters)
        self.setStatList([])
        # train here for num_iter
        trainer = Trainer(self.device, self, model_path='')
        trainer.train(self.data_loaders, num_iters=num_iters, iter_type='batch')
        # Revert to original states here if required.
        self.setCallbacks(orig_cbs)
        return self.scheduler

    def printModules(self, verbose=True):
        # Display names of modules of this model instance. If verbose is True, then
        # only display immediate children modules, otherwise display a nested heirachy
        # of depth 4.
        for name, module in self.pred_fn.named_children():
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


# Given a model (an object inherited from nn.Module), the Trainer class will handle 
# training, hyperparameter tuning, evaluation, logging information.
class Trainer():
    """

    Class attributes:

    model_arch: The model architecture.
    """
    def __init__(self, device, model, model_path):
        self.device = device
        self.pred_fn = model.getPredFn()
        self.scheduler = model.scheduler
        self.criterion = model.getCriterion()
        self.optimizer = model.getOptim()
        self.stat_recorder = model.stat_recorder
        self.display = NotebookDisplay(model)
        if model.callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = model.callbacks
        self.callbacks.append(self.display)
        self.batch_size = model.batch_size
        self.config_name = '{}_{}_'.format(model.pred_fn_class, model.optim_name)

    def train(self, data_loaders, num_iters=1, iter_type='epoch', 
            checkpoint_per_epoch=1, stat_freq_batches=1, update_prog_bar_iter=5,
            verbose=True):
        """
        This method allows training of a model based on a given optimizer and criterion. The training
        can be run for a specified number of epochs (one epoch is full pass over training data) or for
        a specified number of iterations.
        Arguments:
        - iter_type: set to 'epoch' (trains for num_iter epochs) or 'batch' (runs for num_iter batches).
        
        """
        iter_per_epoch = len(data_loaders)
        checkpoint_freq = floor(iter_per_epoch/checkpoint_per_epoch)
        torch.set_grad_enabled(True)
        max_iters = 1e6
        max_epochs = 1e6
        if (iter_type == 'epoch'):
            max_epochs = num_iters
        else:
            max_iters = num_iters
        epoch = 0
        iter_count = 0
        self.config_name += 'train(batch_size={}, iters={} {})'.format(self.batch_size, num_iters, iter_type)
        for cb in self.callbacks: cb.on_train_begin(num_iters=num_iters, iter_type=iter_type)
        while epoch < max_epochs:
            for cb in self.callbacks: cb.on_epoch_begin()
            for iter_num, data in enumerate(data_loaders['train'], 1):
                for cb in self.callbacks: cb.on_batch_begin()
                inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.pred_fn(inputs)
                loss = self.criterion(outputs, labels)
                # backward pass
                loss.backward()
                self.optimizer.step()
                stats_dict = {'train_loss' : loss.item(),
                        'outputs' : outputs,
                        'labels' : labels}
                for cb in self.callbacks: cb.on_batch_end(stats_dict=stats_dict)
                if (iter_num%checkpoint_freq == 0):
                    #checkpoint
                    pass
                iter_count += 1
                if (iter_count == max_iters):
                    break
            if (iter_count == max_iters):
                break
            epoch += 1
            for cb in self.callbacks: cb.on_epoch_end()
        for cb in self.callbacks: cb.on_train_end(config_name=self.config_name)

    def plotLoss(self):
        self.stat_recorder.plotLoss()
    
    def plotLR(self):
        if self.scheduler is not None:
            fig = plt.gcf()
            ax = plt.gca()
            fig.set_size_inches(8,5)
            fig.set_dpi(80)
            ax.set_ylabel('Learning Rate')
            ax.set_xlabel('Iteration')
            _ = plt.plot(np.arange(len(self.scheduler.lr_series)), 
                    np.asarray(self.scheduler.lr_series))


    def SaveCheckpoint(self, model_path):
        state = {'model_dict' : self.pred_fn.state_dict(),
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
