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
    - configs: Configuration dataclass containing training configurations.
    """

    def __init__(self, 
                 dynamics,
                 model,
                 dataloader: DataLoader,
                 configs: dataclass):
    
        #...configs:
        self.configs = configs
        self.dynamics = dynamics
        self.model = model
        self.dataloader = dataloader
        self.workdir = Path(configs.workdir)
        self.epochs = configs.EPOCHS
        self.early_stopping = configs.EPOCHS if configs.early_stopping is None else configs.early_stopping
        self.min_epochs = 0 if configs.min_epochs is None else configs.min_epochs
        self.print_epochs = 1 if configs.print_epochs is None else configs.print_epochs
        self.fix_seed = configs.fix_seed

        #...logger & tensorboard:
        os.makedirs(self.workdir/'tensorboard', exist_ok=True)
        self.writer = SummaryWriter(self.workdir/'tensorboard')  
        self.logger = Logger(configs, self.workdir/'training.log')

    def train(self):

        train = Train_Step()
        valid = Validation_Step()
        optimizer = Optimizer(self.configs)(self.model.parameters())
        scheduler = Scheduler(self.configs)(optimizer)

        #...logging

        self.logger.logfile.info("Training configurations:")
        for field in fields(self.configs): self.logger.logfile.info(f"{field.name}: {getattr(self.configs, field.name)}")
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
            self.model.load_state_dict(torch.load(path/'best_epoch_model.pth', map_location=(torch.device('cpu') if self.configs.DEVICE=='cpu' else None)))
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


from DynGenModels.datamodules.cathode.dataprocess import PreProcessCathodeData

class ClassifierTrainer:

    """
    Trainer for dynamic generative models.
    
    Attributes:
    - dynamics: The model dynamics to train.
    - dataloader: DataLoader providing training and optionally validation data.
    - configs: Configuration dataclass containing training configurations.
    """

    def __init__(self, 
                 classifier,
                 model,
                 dataloader: DataLoader,
                 configs: dataclass):
    
        #...configs:
        self.configs = configs
        self.classifier = classifier
        self.model = model
        self.dataloader = dataloader
        self.workdir = Path(configs.workdir)
        self.epochs = configs.EPOCHS
        self.early_stopping = configs.EPOCHS if configs.early_stopping is None else configs.early_stopping
        self.min_epochs = 0 if configs.min_epochs is None else configs.min_epochs
        self.print_epochs = 1 if configs.print_epochs is None else configs.print_epochs
        self.fix_seed = configs.fix_seed

        #...logger & tensorboard:
        os.makedirs(self.workdir/'tensorboard', exist_ok=True)
        self.writer = SummaryWriter(self.workdir/'tensorboard')  
        self.logger = Logger(configs, self.workdir/'training.log')

    def train(self):

        train = Train_Step()
        valid = Validation_Step()
        optimizer = Optimizer(self.configs)(self.model.parameters())
        scheduler = Scheduler(self.configs)(optimizer)

        #...logging

        self.logger.logfile.info("Classifier training configurations:")
        for field in fields(self.configs): self.logger.logfile.info(f"{field.name}: {getattr(self.configs, field.name)}")
        self.logger.logfile_and_console('number of training parameters: {}'.format(sum(p.numel() for p in self.model.parameters())))
        self.logger.logfile_and_console("start classifier training...")

        #...train

        for epoch in tqdm(range(self.epochs), desc="epochs"):
            train.update(model=self.model, loss_fn=self.classifier.loss, dataloader=self.dataloader.train, optimizer=optimizer) 
            valid.update(model=self.model, loss_fn=self.classifier.loss, dataloader=self.dataloader.valid, seed=self.fix_seed)
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
        
    @torch.no_grad()
    def test(self, test_data):
        
        data = test_data[..., 1:] # remove labels
        ppc = PreProcessCathodeData(data=data, methods=self.configs.preprocess)
        ppc.preprocess()
        data = ppc.features.clone()
        labels = test_data[...,0].unsqueeze(-1)
        
        self.predictions = {}
        probs = self.classifier.predict(model=self.best_epoch_model, batch=data)
        self.predictions['datasets'] = torch.cat([labels, probs], dim=-1) 
        labels = labels.squeeze(-1)
        
        for l in [0,1]:
            self.predictions[l] = self.predictions['datasets'][labels == l][:, -1:]


    def load(self, path: str=None, model: str=None):
        path = self.workdir if path is None else Path(path)
        if model is None:
            self.model.load_state_dict(torch.load(path/'best_classifier_epoch_model.pth'))
            self.best_epoch_model = deepcopy(self.model)
            self.model.load_state_dict(torch.load(path/'last_classifier_epoch_model.pth'))
            self.last_epoch_model = deepcopy(self.model)
        elif model == 'best':
            self.model.load_state_dict(torch.load(path/'best_classifier_epoch_model.pth'))
            self.best_epoch_model = deepcopy(self.model)
        elif model == 'last':
            self.model.load_state_dict(torch.load(path/'last_classifier_epoch_model.pth'))
            self.last_epoch_model = deepcopy(self.model)
        else: raise ValueError("which_model must be either 'best_classifier', 'last_classifier', or None")

    def _save_best_epoch_model(self, improved):
        if improved:
            torch.save(self.model.state_dict(), self.workdir/'best_classifier_epoch_model.pth')
            self.best_epoch_model = deepcopy(self.model)
        else: pass

    def _save_last_epoch_model(self):
        torch.save(self.model.state_dict(), self.workdir/'last_classifier_epoch_model.pth') 
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
        plt.savefig(self.workdir / 'classifier_losses.png')
        plt.close()
