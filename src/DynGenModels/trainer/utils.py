import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass

class Train_Step(nn.Module):

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
            self.loss = loss_current.detach().cpu().numpy()

        self.losses.append(self.loss) 

class Validation_Step(nn.Module):

    def __init__(self, loss_fn, min_epochs=10, print_epochs=5):
        super(Validation_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.print_epoch = print_epochs
        self.min_epochs = min_epochs
        self.losses = []
        
    @torch.no_grad()
    def update(self, dataloader: DataLoader):
        self.loss = 0
        self.epoch += 1

        for batch in dataloader:
            loss_current = self.loss_fn(batch)
            self.loss = loss_current.detach().cpu().numpy()
        self.losses.append(self.loss) 

    @torch.no_grad()
    def checkpoint(self, early_stopping=None):
        terminate = False
        improved = False

        if self.loss < self.loss_min:
            self.loss_min = self.loss
            self.patience = 0
            improved = True 
        else: self.patience += 1 if self.epoch > self.min_epochs else 0

        if self.patience >= early_stopping: 
            terminate = True
        if self.epoch % self.print_epoch == 1:
            print("\t test loss: {}  (min loss: {})".format(self.loss, self.loss_min))

        return terminate, improved


class RNGStateFixer:
    def __init__(self, seed):
        self.seed = seed
        self.saved_rng_state = None
    def __enter__(self):
        self.saved_rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        return 
    def __exit__(self, *args):
        torch.set_rng_state(self.saved_rng_state)
        return
    

class Optimizer:
    def __init__(self, configs: dataclass):
        self.configs = configs

    def get_optimizer(self, parameters):

        optim_args = {'lr': self.configs.lr, 'weight_decay': self.configs.weight_decay}

        if self.configs.optimizer == 'Adam':
            if hasattr(self.configs, 'betas'):
                optim_args['betas'] = self.configs.betas
            if hasattr(self.configs, 'eps'):
                optim_args['eps'] = self.configs.eps
            if hasattr(self.configs, 'amsgrad'):
                optim_args['amsgrad'] = self.configs.amsgrad
            return torch.optim.Adam(parameters, **optim_args)
        
        elif self.configs.optimizer == 'AdamW':
            if hasattr(self.configs, 'betas'):
                optim_args['betas'] = self.configs.betas
            if hasattr(self.configs, 'eps'):
                optim_args['eps'] = self.configs.eps
            if hasattr(self.configs, 'amsgrad'):
                optim_args['amsgrad'] = self.configs.amsgrad
            return torch.optim.AdamW(parameters, **optim_args)
        
        else:
            raise ValueError(f"Unsupported optimizer: {self.configs.optimizer}")
        
    def clip_gradients(self, optimizer):
        if self.configs.gradient_clip:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.configs.gradient_clip)

    def __call__(self, parameters):

        optimizer = self.get_optimizer(parameters)

        #...override the optimizer.step() to include gradient clipping
        original_step = optimizer.step

        def step_with_clipping(closure=None):
            self.clip_gradients(optimizer)
            original_step(closure)
            
        optimizer.step = step_with_clipping
        return optimizer
    

# class Scheduler:
#     def __init__(self, configs: dataclass):
#         self.configs = configs

#     def _get_scheduler(self, optimizer):
#         if self.configs.scheduler == 'CosineAnnealingLR':
#             return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs.T_max)

#         elif self.configs.scheduler == 'StepLR':
#             return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.configs.step_size, gamma=self.configs.gamma)

#         elif self.configs.scheduler == 'ExponentialLR':
#             return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.configs.gamma)

#         else:
#             raise ValueError(f"Unsupported scheduler: {self.configs.scheduler}")

#     def __call__(self, optimizer):
#         return self._get_scheduler(optimizer)

    