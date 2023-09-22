import torch
from torch.utils.data import Dataset
from DynGenModels.datamodules.fermi.format import FormatData
from DynGenModels.datamodules.fermi.preprocess import PreprocessData

class FermiDataset(Dataset):

    def __init__(self, 
                 dir_path: str=None, 
                 cuts: dict=None,
                 preprocess : list=None
                 ):
        
        self.path = dir_path
        self.cuts = cuts
        self.preprocess_methods = preprocess 
        self.summary_statistics = {}
        self.dataset_list = self.get_data()

    def __getitem__(self, idx):
        output = {}
        datasets = self.dataset_list
        galactic_features = self.apply_preprocessing(sample=datasets[idx]) if self.preprocess_methods is not None else datasets[idx] 
        output['target'] = galactic_features
        output['source'] = torch.rand_like(galactic_features)
        return output

    def __len__(self):
        return self.dataset_list.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_data(self):
        print("INFO: loading and preprocessing data...")
        dataset = torch.tensor(np.load(self.path))
        dataset = self.apply_formatting(dataset)
        print(dataset.shape)
        print('\t- dataset: {} \n \t- shape: {}'.format(self.path, dataset.shape))
        self.summary_statistics['dataset'] = self.summary_stats(dataset)
        return dataset

    def apply_formatting(self, sample):
        sample = FormatData(sample, cuts=self.cuts)
        sample.format()
        return sample.data
    
    def apply_preprocessing(self, sample):
        sample = PreprocessData(data=sample, 
                                stats=self.summary_statistics['dataset'],
                                methods = self.preprocess_methods
                                )
        sample.preprocess()
        return sample.galactic_features
    
    def summary_stats(self, data):
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        min, _ = torch.min(data, dim=0)
        max, _ = torch.max(data, dim=0)
        return (mean, std, min, max)
