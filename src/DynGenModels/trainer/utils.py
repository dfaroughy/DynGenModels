import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, fields

class Train_Step(nn.Module): 

    """
    Represents a single training step.
    
    Attributes:
    - loss_fn: The loss function to use during training.
    """

    def __init__(self, loss_fn):
        super(Train_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.losses = []

    def update(self, dataloader: DataLoader, optimizer):
        self.loss = 0
        self.epoch += 1
        for batch in dataloader:
            optimizer.zero_grad()
            loss_current = self.loss_fn(batch)
            loss_current.backward()
            optimizer.step()  
            self.loss += loss_current.detach().cpu().numpy() / len(dataloader)
        self.losses.append(self.loss) 

class Validation_Step(nn.Module):

    """
    Represents a validation step.
    
    Attributes:
    - loss_fn: Loss function to compute validation loss.
    - min_epochs: Minimum number of epochs before considering early stopping.
    """

    def __init__(self, loss_fn, min_epochs=10):
        super(Validation_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.loss_min = np.inf
        self.min_epochs = min_epochs
        self.losses = []
        
    @torch.no_grad()
    def update(self, dataloader: DataLoader, seed=None):
        self.epoch += 1
        self.loss = 0
        self.validate = bool(dataloader)
        if self.validate:
            with RNGStateFixer(seed):
                for batch in dataloader:
                    loss_current = self.loss_fn(batch)
                    self.loss += loss_current.detach().cpu().numpy() / len(dataloader)
            self.losses.append(self.loss) 

    @torch.no_grad()
    def checkpoint(self, early_stopping=None):
        TERMINATE = False
        IMPROVED = False
        if self.validate:
            if self.loss < self.loss_min:
                self.loss_min = self.loss
                self.patience = 0
                IMPROVED = True 
            else: self.patience += 1 if self.epoch > self.min_epochs else 0
            if self.patience >= early_stopping: TERMINATE = True
        return TERMINATE, IMPROVED


class Optimizer:

    """
    Custom optimizer class with support for gradient clipping.
    
    Attributes:
    - configs: Configuration dataclass containing optimizer configurations.
    """

    def __init__(self, configs: dataclass):
        self.configs = configs

    def get_optimizer(self, parameters):

        optim_args = {'lr': self.configs.lr, 'weight_decay': self.configs.weight_decay}

        if self.configs.optimizer == 'Adam':
            if hasattr(self.configs, 'betas'): optim_args['betas'] = self.configs.optimizer_betas
            if hasattr(self.configs, 'eps'): optim_args['eps'] = self.configs.optimizer_eps
            if hasattr(self.configs, 'amsgrad'): optim_args['amsgrad'] = self.configs.optimizer_amsgrad
            return torch.optim.Adam(parameters, **optim_args)
        
        elif self.configs.optimizer == 'AdamW':
            if hasattr(self.configs, 'betas'): optim_args['betas'] = self.configs.optimizer_betas
            if hasattr(self.configs, 'eps'): optim_args['eps'] = self.configs.optimizer_eps
            if hasattr(self.configs, 'amsgrad'): optim_args['amsgrad'] = self.configs.optimizer_amsgrad
            return torch.optim.AdamW(parameters, **optim_args)
        
        else:
            raise ValueError(f"Unsupported optimizer: {self.configs.optimizer}")
        
    def clip_gradients(self, optimizer):
        if self.configs.gradient_clip: torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.configs.gradient_clip)

    def __call__(self, parameters):
        optimizer = self.get_optimizer(parameters)
        #...override the optimizer.step() to include gradient clipping
        original_step = optimizer.step

        def step_with_clipping(closure=None):
            self.clip_gradients(optimizer)
            original_step(closure)          
        optimizer.step = step_with_clipping
        return optimizer

class Scheduler:

    """
    Custom scheduler class to adjust the learning rate during training.
    
    Attributes:
    - configs: Configuration dataclass containing scheduler configurations.
    """

    def __init__(self, configs: dataclass):
        self.configs = configs

    def get_scheduler(self, optimizer):
        if self.configs.scheduler == 'CosineAnnealingLR': return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs.scheduler_T_max)
        elif self.configs.scheduler == 'StepLR': return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.configs.scheduler_step_size, gamma=self.configs.scheduler_gamma)
        elif self.configs.scheduler == 'ExponentialLR': return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.configs.scheduler_gamma)
        elif self.configs.scheduler is None: return NoScheduler(optimizer)
        else: raise ValueError(f"Unsupported scheduler: {self.configs.scheduler}")

    def __call__(self, optimizer):
        return self.get_scheduler(optimizer)

class NoScheduler:
    def __init__(self, optimizer): pass
    def step(self): pass    


class RNGStateFixer:
    """
    Context manager to fix the RNG state using a given seed.
    Restores the original state after exiting the context.
    
    Attributes:
    - seed: Seed for the RNG.
    """

    def __init__(self, seed):
        self.seed = seed
        self.saved_rng_state = None
    def __enter__(self):
        if self.seed is not None: 
            self.saved_rng_state = torch.get_rng_state()
            torch.manual_seed(self.seed)
        return     
    def __exit__(self, *args):
        if self.seed is not None: torch.set_rng_state(self.saved_rng_state)
        return


class Logger:
    ''' Logging handler for training and validation.
    '''
    def __init__(self, configs: dataclass, path: Path):
        self.path = path
        self.fh = None  
        self.ch = None 
        self._training_loggers()
        self.logfile.info("Training configurations:")
        for field in fields(configs):
            self.logfile.info(f"{field.name}: {getattr(configs, field.name)}")

    def _training_loggers(self):
        
        self.logfile = logging.getLogger('file_logger')
        self.logfile.setLevel(logging.INFO)
        self.fh = logging.FileHandler(self.path) 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.logfile.addHandler(self.fh)
        self.logfile.propagate = False 
        
        self.console = logging.getLogger('console_logger')
        self.console.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()  
        ch_formatter = logging.Formatter('%(message)s') 
        self.ch.setFormatter(ch_formatter)
        self.console.addHandler(self.ch)
        self.console.propagate = False 

    def logfile_and_console(self, message):
        self.logfile.info(message)
        self.console.info(message)
    
    

    def close(self):
        if self.fh:
            self.fh.close()
            self.logfile.removeHandler(self.fh)
        if self.ch:
            self.ch.close()
            self.console.removeHandler(self.ch)