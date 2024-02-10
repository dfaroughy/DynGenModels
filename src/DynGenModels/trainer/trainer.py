import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from dataclasses import dataclass, fields
from copy import deepcopy
import os

from DynGenModels.trainer.utils import Train_Step, Validation_Step, Optimizer, Scheduler, Logger

class DynGenModelTrainer:

    """
    Trainer for dynamic generative models.
    
    Attributes:
    - dynamics: The model dynamics to train.
    - dataloader: DataLoader providing training and optionally validation data.
    - config: Configuration dataclass containing training configurations.
    """

    def __init__(self, 
                 dynamics,
                 model,
                 dataloader: DataLoader,
                 config: dataclass):
    
        #...config:
        self.config = config
        self.dynamics = dynamics
        self.model = model
        self.dataloader = dataloader
        self.workdir = Path(config.WORKDIR)
        self.epochs = config.EPOCHS
        self.early_stopping = config.EPOCHS if config.EARLY_STOPPING is None else config.EARLY_STOPPING
        self.min_epochs = 0 if config.MIN_EPOCHS is None else config.MIN_EPOCHS
        self.print_epochs = 1 if config.PRINT_EPOCHS is None else config.PRINT_EPOCHS
        self.fix_seed = config.FIX_SEED

        #...logger & tensorboard:
        os.makedirs(self.workdir/'tensorboard', exist_ok=True)
        self.writer = SummaryWriter(self.workdir/'tensorboard')  
        self.logger = Logger(config, self.workdir/'training.log')

    def train(self):

        train = Train_Step()
        valid = Validation_Step()
        optimizer = Optimizer(self.config)(self.model.parameters())
        scheduler = Scheduler(self.config)(optimizer)

        #...logging

        self.logger.logfile.info("Training configurations:")
        for field in fields(self.config): self.logger.logfile.info(f"{field.name}: {getattr(self.config, field.name)}")
        self.logger.logfile_and_console('number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.logger.logfile_and_console("start training...")

        #...train

        for epoch in tqdm(range(self.epochs), desc="epochs"):
            train.update(model=self.model, loss_fn=self.dynamics.loss, dataloader=self.dataloader.train, optimizer=optimizer) 
            valid.update(model=self.model, loss_fn=self.dynamics.loss, dataloader=self.dataloader.valid, seed=self.fix_seed)
            TERMINATE, IMPROVED = valid.checkpoint(min_epochs=self.min_epochs, early_stopping=self.early_stopping)
            scheduler.step() 
            self._log_losses(train, valid, epoch)
            self._save_best_epoch_model(IMPROVED)
            
            if TERMINATE: 
                stop_message = "early stopping triggered! Reached maximum patience at {} epochs".format(epoch)
                self.logger.logfile_and_console(stop_message)
                break
            
        self._save_last_epoch_model()
        self._save_best_epoch_model(not bool(self.dataloader.valid)) # best = last epoch if there is no validation, needed as a placeholder for pipeline
        self.plot_loss(valid_loss=valid.losses, train_loss=train.losses)
        self.logger.close()
        self.writer.close() 
        
    def load(self, path: str=None, model: str=None):
        path = self.workdir if path is None else Path(path)
        if model is None:
            self.model.load_state_dict(torch.load(path/'best_epoch_model.pth'))
            self.best_epoch_model = deepcopy(self.model)
            self.model.load_state_dict(torch.load(path/'last_epoch_model.pth'))
            self.last_epoch_model = deepcopy(self.model)
        elif model == 'best':
            self.model.load_state_dict(torch.load(path/'best_epoch_model.pth', map_location=(torch.device('cpu') if self.config.DEVICE=='cpu' else None)))
            self.best_epoch_model = deepcopy(self.model)
        elif model == 'last':
            self.model.load_state_dict(torch.load(path/'last_epoch_model.pth'))
            self.last_epoch_model = deepcopy(self.model)
        else: raise ValueError("which_model must be either 'best', 'last', or None")

    def _save_best_epoch_model(self, improved):
        if improved:
            torch.save(self.model.state_dict(), self.workdir/'best_epoch_model.pth')
            self.best_epoch_model = deepcopy(self.model)
        else: pass

    def _save_last_epoch_model(self):
        torch.save(self.model.state_dict(), self.workdir/'last_epoch_model.pth') 
        self.last_epoch_model = deepcopy(self.model)

    def _log_losses(self, train, valid, epoch):
        self.writer.add_scalar('Loss/train', train.loss, epoch)
        self.writer.add_scalar('Loss/valid', valid.loss, epoch)
        message = "\tEpoch: {}, train loss: {}, valid loss: {}  (min valid loss: {})".format(epoch, train.loss, valid.loss, valid.loss_min)
        self.logger.logfile.info(message)
        if epoch % self.print_epochs == 1:            
            self.plot_loss(valid_loss=valid.losses, train_loss=train.losses)
            self.logger.console.info(message)

    def plot_loss(self, valid_loss, train_loss):
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(range(len(valid_loss)), np.array(valid_loss), color='r', lw=1, linestyle='-', label='Validation')
        ax.plot(range(len(train_loss)), np.array(train_loss), color='b', lw=1, linestyle='--', label='Training', alpha=0.8)
        ax.set_xlabel("Epochs", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.set_title("Training & Validation Loss Over Epochs", fontsize=6)
        ax.set_yscale('log')
        ax.legend(fontsize=6)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        plt.savefig(self.workdir / 'losses.png')
        plt.close()