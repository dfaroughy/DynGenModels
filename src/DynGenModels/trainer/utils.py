import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class Train_Step(nn.Module):

    def __init__(self, loss_fn, gradient_clip=None):
        super(Train_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.losses = []
        self.gradient_clip = gradient_clip

    def update(self, dataloader: DataLoader, optimizer):
        self.loss = 0
        self.epoch += 1

        for batch in dataloader:
            optimizer.zero_grad()
            loss_current = self.loss_fn(batch)
            loss_current.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.gradient_clip)
            optimizer.step()  
            self.loss += loss_current.detach().cpu().numpy()
        self.losses.append(self.loss) 

class Validation_Step(nn.Module):

    def __init__(self, loss_fn, warmup_epochs=10, print_epochs=5):
        super(Validation_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.print_epoch = print_epochs
        self.warmup_epochs = warmup_epochs
        self.losses = []
        
    @torch.no_grad()
    def update(self, dataloader: DataLoader):
        self.loss = 0
        self.epoch += 1
        for batch in dataloader:
            loss_current = self.loss_fn(batch)
            self.loss += loss_current.detach().cpu().numpy()
        # self.loss = self.loss / len(dataloader.dataset)
        self.losses.append(self.loss) 

    @torch.no_grad()
    def checkpoint(self, early_stopping=None):
        terminate = False
        improved = False
        
        # if early_stopping is not None:     
        if self.loss < self.loss_min:
            self.loss_min = self.loss
            self.patience = 0
            improved = True 

        else: self.patience += 1 if self.epoch > self.warmup_epochs else 0

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