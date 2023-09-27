import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from dataclasses import dataclass

from DynGenModels.trainer.utils import Train_Step, Validation_Step

class FlowMatchTrainer(nn.Module):

    def __init__(self, 
                 dynamics,
                 dataloader: DataLoader,
                 config: dataclass):
    
        self.dynamics = dynamics
        self.dataloader = dataloader
        self.workdir = config.workdir
        self.lr = config.lr
        self.epochs = config.epochs
        self.early_stopping = config.early_stopping 
        self.warmup_epochs = config.warmup_epochs
        self.print_epochs = config.print_epochs
        self.seed = config.seed

        os.makedirs(self.workdir+'/tensorboard', exist_ok=True)
        self.writer = SummaryWriter(self.workdir+'/tensorboard')  # tensorboard writer

    def train(self):
        train = Train_Step(loss_fn=self.dynamics.loss)
        valid = Validation_Step(loss_fn=self.dynamics.loss, 
                                warmup_epochs=self.warmup_epochs, 
                                print_epochs=self.print_epochs)
        optimizer = torch.optim.Adam(self.dynamics.net.parameters(), lr=self.lr)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        print('INFO: number of training parameters: {}'.format(sum(p.numel() for p in self.dynamics.net.parameters())))
        
        for epoch in tqdm(range(self.epochs), desc="epochs"):
            train.update(dataloader=self.dataloader.train, optimizer=optimizer)       
            valid.update(dataloader=self.dataloader.valid)
            scheduler.step() 
            self.writer.add_scalar('Loss/train', train.loss, epoch)
            self.writer.add_scalar('Loss/valid', valid.loss, epoch)

            if valid.stop(save_best=self.dynamics.net,
                              early_stopping=self.early_stopping, 
                              workdir=self.workdir): 
                print("INFO: early stopping triggered! Reached maximum patience at {} epochs".format(epoch))
                break          
        self.writer.close() 

    # def load_state(self, path=None):
    #     path = self.workdir + '/best_model.pth' if path is None else path
    #     self.dynamics.net.load_state_dict(torch.load(path))
