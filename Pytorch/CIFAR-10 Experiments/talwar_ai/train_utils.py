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
from torch.optim.lr_scheduler import StepLR, LambdaLR

class NNModel():
    def __init__(self, model_name, optim_name, criterion_name, num_classes, pre_trained=False):
        self.model_name = model_name
        self.optim_name = optim_name
        self.criterion_name = criterion_name
        self.scheduler = None
        self.num_classes = num_classes
        self.pre_trained = pre_trained
        self.model = self.loadModel()
        self.optimizer = self.loadOptim(self.model.parameters())
        self.criterion = self.loadCriterion()

    def loadModel(self):
        """
        Loads model architecture along with pre-trained parameters (if asked). Replaces
        plain vanilla pooling with adaptive pooling, and updates the number of output
        nodes in the final layer.

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

    def loadOptim(self, params):
        """
        Initializes the optimizer for this model. By default, all model parameters
        are passed to the optimizer - this can be modified by using the freezeParams()
        method.

        """
        if (self.optim_name == 'sgd'):
            optimizer = optim.SGD(params, lr=0.1)
        elif (self.optim_name == 'adam'):
            optimizer = optim.Adam(params, lr=0.1)
        else:
            pass
        return optimizer
    
    def loadCriterion(self):
        if (self.criterion_name == 'cross_entropy'):
            criterion = nn.CrossEntropyLoss()
        else:
            pass
        return criterion

    def freezeParams(self, freeze_modules=[]):
        """
        Freezes (sets requires_grad=FalsE) parameters of the model, based on arguments 
        provided:
        (1) freeze_modules: list of module names - all params belonging to these
            modules are frozen.

        """
        # First pass to set requires_grad=True for all parameters.
        for param in self.model.parameters():
            param.requires_grad = True
        # Iterate over freeze_modules to set their requires_grad=False
        for name, module in self.model.named_modules():
            if (name in freeze_modules):
                for param in module.parameters():
                    param.requires_grad = False
        # Recreate optimizer with updated parameters
        self.optimizer = self.loadOptim(filter(lambda p: p.requires_grad, self.model.parameters()))

    def exploreLR(self, num_iter=500, min_lr=1e-4, max_lr=10.0, mult_factor=1.1):
        # First save state of the existing optimizer, change its learning rate to min_lr,
        # and revert to original state in the end.
        orig_state_dict = self.optimizer.state_dict
        for param in self.optimizer.param_groups:
            param['lr'] = 1.0
        arg_dict = {'init_lr' : min_lr,
                'final_lr' : max_lr,
                'num_iter' : num_iter,
                'mult' : mult_factor}
        self.setScheduler('linear', arg_dict)
        # train here for num_iter
        # revert to original state here

    def setScheduler(self, sched_name, arg_dict):
        if (sched_name == 'linear'):
            lambda_fn = lambda batch: min(arg_dict['final_lr'], 
                    arg_dict['init_lr']*arg_dict['mult']**batch)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_fn)  
        else:
            pass

    def printModules(self, verbose=True):
        # Display names of modules of this model instance. If verbose is True, then
        # only display immediate children modules, otherwise display a nested heirachy
        # of depth 4.
        for name, module in self.model.named_children():
            print('Module: ' + name)
            if not verbose:
                for name1, module1 in module.named_children():
                    print('..' + name1)
                    for name2, module2 in module1.named_children():
                        print('....' + name2)
                        for name3, module3 in module2.named_children():
                            print('......' + name3)
                            for name4, module4 in module3.named_children():
                                print('........' + name4)


# Given a model (an object inherited from nn.Module), this model will handle 
# training, hyperparameter tuning, evaluation, logging information.
class Trainer():
    def __init__(self, device, model, criterion, optimizer, scheduler, model_path, sched_type='batch'):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
    def train(self, data_loaders, num_epochs=1, use_iterations=False, num_iters=500, 
            checkpoint_per_epoch=1, eval_stats=['loss'], stat_per_epoch=1, update_prog_bar_iter=5,
            stop_at_iter=-1, verbose=True):
        progress_bar = tqdm(total=num_epochs)
        iter_per_epoch = len(data_loaders)
        stat_freq = floor(iter_per_epoch/stat_per_epoch)
        checkpoint_freq = floor(iter_per_epoch/checkpoint_per_epoch)
        running_loss = 0.0
        torch.set_grad_enabled(True)
        iter_count = 0
        for epoch in range(num_epochs):
            for i, data in enumerate(data_loaders['train'], 1):
                if verbose and (i%update_prog_bar_iter == 0):
                    progress_bar.set_description('e {} i {}'.format(epoch + 1, i))
                inputs, labels = data['image'].to(self.device), data['label'].long().to(self.device)
                # forward pass
                if (sched_type == 'batch'):
                    self.scheduler.step()
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
                    valid_stats = Trainer._evalStats(self.device, self.model, self.criterion, 
                            data_loaders['valid'], eval_stats=eval_stats)
                    train_loss = running_loss/stat_freq
                    running_loss = 0.0
                    progress_bar.set_postfix(train_loss='{:.2f}'.format(train_loss), 
                                         val_loss='{:.2f}'.format(valid_stats['loss']))
                if (i%checkpoint_freq == 0):
                    #checkpoint
                    pass
                iter_count += 1
                if (iter_count == stop_at_iter):
                    break
            if verbose: progress_bar.update(1)
            if (iter_count == stop_at_iter):
                break

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
