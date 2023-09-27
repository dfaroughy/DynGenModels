import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

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
            self.loss += loss_current.detach().cpu().numpy()

        self.loss = self.loss / len(dataloader.dataset)
        self.losses.append(self.loss) 

class Validation_Step(nn.Module):

    def __init__(self, loss_fn, warmup_epochs=10, print_epochs=5):
        super(Validation_Step, self).__init__()
        self.loss_fn = loss_fn
        self.loss = 0
        self.epoch = 0
        self.patience = 0
        self.loss_min = np.inf
        self.terminate_loop = False
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

        self.loss = self.loss / len(dataloader.dataset)
        self.losses.append(self.loss) 

    @torch.no_grad()
    def stop(self, save_best, early_stopping, workdir):
        if early_stopping is not None:
            if self.loss < self.loss_min:
                self.loss_min = self.loss
                self.patience = 0
                torch.save(save_best.state_dict(), workdir + '/best_model.pth')  
            else: self.patience += 1 if self.epoch > self.warmup_epochs else 0
            if self.patience >= early_stopping: 
                self.terminate_loop = True
                torch.save(save_best.state_dict(), workdir + '/last_epoch_model.pth')
        else:
            torch.save(save_best.state_dict(), workdir + '/last_epoch_model.pth')
        if self.epoch % self.print_epoch == 1:
            print("\t test loss: {}  (min loss: {})".format(self.loss, self.loss_min))
        return self.terminate_loop

