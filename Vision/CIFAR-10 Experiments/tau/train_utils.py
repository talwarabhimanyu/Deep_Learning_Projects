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

    Class Attributes:

    pred_fn_class: the Prediction Function Class to be used. This will be one of 
    pre-trained Prediction Function Classes such as 'resnet18'.

    optimizer: initialized to SGD with lr=0.01 for all model parameters. This can
    be modified via loadOptim()

    Class Methods:

    loadOptim(optim_name, params): Loads an optimizer of type optim_name and initializes
    it using params.
        optim_name: String. One of 'sgd', 'adam'.
        
        params: It is of type parameters or a dictionary specifying parameter groups with or
        without group lrs.

        callbacks: This will be initialized with the Neural Net's clock and its stat_recorder. 
        These two callback must never be removed from the callbacks list. Additional
        callbacks can be added to the list using addCallbacks() method.

    """
    def __init__(self, device, pred_fn_class, criterion_name, num_classes, data_loaders, pre_trained=False):
        self.device = device
        self.pred_fn_class = pred_fn_class
        self.criterion_name = criterion_name
        self.scheduler = None
        self.num_classes = num_classes
        self.data_loaders = data_loaders
        self.pre_trained = pre_trained
        self.model = self.loadModel()
        self.optimizer = self.loadOptim('default', self.model.parameters())
        self.criterion = self.loadCriterion()
        self.clock = Clock()
        self.stat_recorder = StatRecorder(self.device, self.clock, self.model, self.criterion, self.data_loaders)
        self.callbacks = [self.clock, self.stat_recorder]

    def loadModel(self):
        """
        Loads model architecture along with pre-trained parameters (if asked). Replaces
        plain vanilla pooling with adaptive pooling for resnet models, and sets the 
        number of output nodes in the final layer equal to number of classes.

        """
        if (self.pred_fn_class == 'resnet18'):
            base_model = models.resnet18(pretrained=self.pre_trained)
        elif (self.pred_fn_class == 'resnet34'):
            base_model = models.resnet34(pretrained=self.pre_trained)
        elif (self.pred_fn_class == 'resnet50'):
            base_model = models.resnet50(pretrained=self.pre_trained)
        else:
            pass
        if ('resnet' in self.pred_fn_class):
            base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            base_model.fc = nn.Linear(base_model.fc.in_features, self.num_classes)
        base_model = base_model.to(device)
        return base_model

    def resetModel(self):
        self.model = self.loadModel()
        self.clock.resetClock()
    
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

    def setOptim(self, optim_name, optim_params=None, **kwargs):
        if optim_params is None:
            optim_params = self.model.parameters()
        self.optimizer = self.loadOptim(optim_name, optim_params, **kwargs)

    def getOptim(self):
        return self.optimizer

    def setScheduler(self, sched_name, **kwargs):
        if sched_name == 'finder':
            self.scheduler = LR_Finder(self.optimizer, self.clock, **kwargs)
        elif sched_name == 'cyclical':
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

    def exploreLR(self, num_iters=200, min_lr=1e-4, max_lr=10.0):
        # Save current state of model and optimizer - they can be reverted to
        # original states if required.
        orig_optim_state_dict = self.optimizer.state_dict
        orig_model_state_dict = self.model.state_dict()
        orig_cbs = self.callbacks
        self.resetModel()
        self.setOptim('sgd', self.model.parameters())
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


# Given a model (an object inherited from nn.Module), the Trainer class will handle 
# training, hyperparameter tuning, evaluation, logging information.
class Trainer():
    """

    Class attributes:

    model_arch: The model architecture.
    """
    def __init__(self, device, model_setup, model_path):
        self.device = device
        self.model = model_setup.getModel()
        self.scheduler = model_setup.scheduler
        self.criterion = model_setup.getCriterion()
        self.optimizer = model_setup.getOptim()
        self.stat_recorder = model_setup.stat_recorder
        self.display = NotebookDisplay(model_setup)
        if model_setup.callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = model_setup.callbacks
        self.callbacks.append(self.display)
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

        for cb in self.callbacks: cb.on_train_begin(num_iters=num_iters, iter_type=iter_type)
        while epoch < max_epochs:
            for cb in self.callbacks: cb.on_epoch_begin()
            for iter_num, data in enumerate(data_loaders['train'], 1):
                for cb in self.callbacks: cb.on_batch_begin()
                inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
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
