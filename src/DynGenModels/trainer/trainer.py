import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from dataclasses import dataclass
import copy
from copy import deepcopy

from DynGenModels.trainer.utils import Train_Step, Validation_Step, RNGStateFixer, Optimizer

class DynGenModelTrainer:

    def __init__(self, 
                 dynamics,
                 dataloader: DataLoader,
                 configs: dataclass):
    
        self.configs = configs
        self.dynamics = dynamics
        self.dataloader = dataloader
        self.workdir = configs.workdir
        self.epochs = configs.EPOCHS
        self.validate = bool(dataloader.valid)
        self.early_stopping = configs.EPOCHS if configs.early_stopping is None else configs.early_stopping
        self.min_epochs = 0 if configs.min_epochs is None else configs.min_epochs
        self.print_epochs = 1 if configs.print_epochs is None else configs.print_epochs
        self.seed = configs.seed

        os.makedirs(self.workdir+'/tensorboard', exist_ok=True)
        self.writer = SummaryWriter(self.workdir+'/tensorboard')  # tensorboard writer

    def train(self):

        train = Train_Step(loss_fn=self.dynamics.loss)
        valid = Validation_Step(loss_fn=self.dynamics.loss, min_epochs=self.min_epochs, print_epochs=self.print_epochs)
        optimizer = Optimizer(self.configs)(self.dynamics.net.parameters())

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.dynamics.net.parameters())))
        
        for epoch in tqdm(range(self.epochs), desc="epochs"):
            train.update(dataloader=self.dataloader.train, optimizer=optimizer) 

            if self.validate: 
                with RNGStateFixer(self.seed):
                    valid.update(dataloader=self.dataloader.valid)
                    terminate, improved = valid.checkpoint(early_stopping=self.early_stopping)
                    if improved:
                        torch.save(self.dynamics.net.state_dict(), self.workdir + '/best_epoch_model.pth')
                        self.best_epoch_model = deepcopy(self.dynamics.net)
                    if terminate: 
                        print("INFO: early stopping triggered! Reached maximum patience at {} epochs".format(epoch))
                        break  
                    self.writer.add_scalar('Loss/valid', valid.loss, epoch)

            # scheduler.step() 
            self.writer.add_scalar('Loss/train', train.loss, epoch)

        torch.save(self.dynamics.net.state_dict(), self.workdir + '/last_epoch_model.pth') 
        self.last_epoch_model = deepcopy(self.dynamics.net) 
        if self.validate is False:
            self.best_epoch_model = deepcopy(self.dynamics.net)  
        self.writer.close() 

    def load(self, path: str=None):
        path = self.workdir if path is None else path
        self.dynamics.net.load_state_dict(torch.load(path + '/best_epoch_model.pth'))
        self.best_epoch_model = self.dynamics.net
        self.dynamics.net.load_state_dict(torch.load(path + '/last_epoch_model.pth'))
        self.last_epoch_model = self.dynamics.net
