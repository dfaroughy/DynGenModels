import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from DynGenModels.trainer.utils import Train_Step, Validation_Step

class FlowMatchTrainer(nn.Module):

    def __init__(self, 
                 dynamics,
                 dataloader: DataLoader,
                 epochs: int=100, 
                 lr: float=0.001, 
                 early_stopping : int=10,
                 warmup_epochs: int=3,
                 workdir: str='./',
                 seed=12345):
    
        self.dynamics = dynamics
        self.dataloader = dataloader
        self.workdir = workdir
        self.lr = lr
        self.seed = seed
        self.early_stopping = early_stopping 
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        os.makedirs(self.workdir+'/tensorboard', exist_ok=True)
        self.writer = SummaryWriter(self.workdir+'/tensorboard')  # tensorboard writer

    def train(self):
        train = Train_Step(loss_fn=self.dynamics.loss)
        valid = Validation_Step(loss_fn=self.dynamics.loss, warmup_epochs=self.warmup_epochs)
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