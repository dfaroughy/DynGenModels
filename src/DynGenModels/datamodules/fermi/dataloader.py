
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from DynGenModels.datamodules.jetnet.datasets import FermiDataset

class FermiDataLoader:

    def __init__(self, 
                 datasets: FermiDataset, 
                 data_split_fracs: list=[0.7, 0.2, 0.1],
                 batch_size: int=1024 
                 ):

        self.datasets = datasets        
        self.data_split_fracs = data_split_fracs
        self.batch_size = batch_size
        self.dataloader()

    def train_val_test_split(self, dataset, train_frac, valid_frac, shuffle=False):
        assert sum(self.data_split_fracs) - 1.0 < 1e-3, "Split fractions do not sum to 1!"
        total_size = len(dataset)
        train_size = int(total_size * train_frac)
        valid_size = int(total_size * valid_frac)
        
        #...define splitting indices

        idx = torch.randperm(total_size) if shuffle else torch.arange(total_size)
        idx_train = idx[:train_size]
        idx_valid = idx[train_size : train_size + valid_size]
        idx_test = idx[train_size + valid_size :]
        
        #...Create Subset for each split

        train_set = Subset(dataset, idx_train)
        valid_set = Subset(dataset, idx_valid)
        test_set = Subset(dataset, idx_test)

        return train_set, valid_set, test_set


    def dataloader(self):

        print("INFO: building dataloaders...")

        #...get training / validation / test samples   

        print("INFO: train/val/test split ratios: {}/{}/{}".format(self.data_split_fracs[0], 
                                                                   self.data_split_fracs[1], 
                                                                   self.data_split_fracs[2]))
        
        train, valid, test = self.train_val_test_split(dataset=self.datasets, 
                                                       train_frac=self.data_split_fracs[0], 
                                                       valid_frac=self.data_split_fracs[1], 
                                                       shuffle=True)

        #...create dataloaders

        self.train = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
        self.valid = DataLoader(dataset=valid,  batch_size=self.batch_size, shuffle=False)
        self.test = DataLoader(dataset=test,  batch_size=self.batch_size, shuffle=True)

        print('INFO: train size: {}, validation size: {}, testing sizes: {}'.format(len(self.train.dataset), 
                                                                                    len(self.valid.dataset), 
                                                                                    len(self.test.dataset)))
