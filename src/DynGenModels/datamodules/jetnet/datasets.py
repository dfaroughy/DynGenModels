import torch
import os
import h5py
import json
from torch.utils.data import Dataset
from DynGenModels.datamodules.jetnet.format import FormatData
from DynGenModels.datamodules.jetnet.preprocess import PreprocessData

class JetNetDataset(Dataset):

    def __init__(self, 
                 dir_path: str=None, 
                 datasets: dict=None,
                 class_labels: dict=None,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 preprocess : list=None,
                 max_num_jets: int=None,
                 max_num_constituents: int=150,
                 remove_negative_pt: bool=False):
        
        self.path = dir_path
        self.datasets = datasets
        self.class_labels = class_labels
        self.max_num_jets = max_num_jets
        self.max_num_constituents = max_num_constituents
        self.particle_features = particle_features
        self.remove_negative_pt = remove_negative_pt
        self.preprocess_methods = preprocess 
        self.summary_statistics = {}
        self.dataset_list = self.get_data()

    def __getitem__(self, idx):
        output = {}
        datasets, labels = self.dataset_list
        output['label'] = labels[idx]
        output['mask'] = datasets[idx][:, -1]
        particles, jet = self.apply_preprocessing(sample=datasets[idx]) if self.preprocess_methods is not None else (datasets[idx], None) 
        output['target'] = particles
        output['source'] = torch.rand_like(particles)
        if jet is not None: output['jet_features'] = jet 
        return output

    def __len__(self):
        return self.dataset_list[0].size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_data(self):
        print("INFO: loading and preprocessing data...")
        data_list, label_list = [], []

        for data in list(self.datasets.keys()):
            file_name = self.datasets[data][0]
            key = self.datasets[data][1] if len(self.datasets[data]) > 1 else None
            file_path = os.path.join(self.path, file_name)

            with h5py.File(file_path, 'r') as f:
                label = self.class_labels[data] if self.class_labels is not None else None
                dataset = torch.from_numpy(f[key][...])
                dataset = self.apply_formatting(dataset)
                self.summary_statistics[label] = self.summary_stats(dataset)
                data_list.append(dataset)
                label_list.append(torch.full((dataset.shape[0],), label))
                print('\t- {} {}: {}  [{}, {}]  shape: {}'.format('test' if label==-1 else 'model', '' if label==-1 else label, data, file_name, key, dataset.shape))
                
        data_tensor = torch.cat(data_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0) 
        self.summary_statistics['dataset'] = self.summary_stats(data_tensor)
        return data_tensor, label_tensor

    def apply_formatting(self, sample):
        sample = FormatData(sample,
                            max_num_jets=self.max_num_jets,
                            max_num_constituents=self.max_num_constituents,
                            particle_features=self.particle_features,
                            remove_negative_pt=self.remove_negative_pt)
        sample.format()
        return sample.data
    
    def apply_preprocessing(self, sample):
        sample = PreprocessData(data=sample, 
                                stats=self.summary_statistics['dataset'],
                                methods = self.preprocess_methods
                                )
        sample.preprocess()
        return sample.particle_features, sample.jet_features
    
    def summary_stats(self, data):
        data_flat = data_flat = data.reshape(-1, data.shape[-1])  # data.view(-1, data.shape[-1])
        mask = data_flat[:, -1].bool()
        mean = torch.mean(data_flat[mask],dim=0)
        std = torch.std(data_flat[mask],dim=0)
        min, _ = torch.min(data_flat[mask],dim=0)
        max, _ = torch.max(data_flat[mask],dim=0)
        mean, std, min, max = mean[:-1], std[:-1], min[:-1], max[:-1]
        return (mean, std, min, max)
    
    def save(self, path):
        print("INFO: saving dataset to {}".format(path))
        torch.save(self.dataset_list, os.path.join(path, 'dataset.pth'))

        dataset_args = {'dir_path': self.path, 
                     'datasets': self.datasets, 
                     'class_labels': self.class_labels, 
                     'particle_features': self.particle_features,
                     'preprocess': self.preprocess, 
                     'max_num_jets': self.max_num_jets, 
                     'max_num_constituents': self.max_num_constituents, 
                     'remove_negative_pt': self.remove_negative_pt}
        
        with open(path+'/dataset_configs.json', 'w') as json_file:
            json.dump(dataset_args, json_file, indent=4)

    @staticmethod
    def load(path):
        print("INFO: loading dataset from {}".format(path))
        dataset_list = torch.load(os.path.join(path, 'dataset.pth'))
        with open(path+'/dataset_configs.json') as json_file:
            dataset_args = json.load(json_file)
            loaded_dataset = JetNetDataset(**dataset_args)
        loaded_dataset.dataset_list = dataset_list
        return loaded_dataset
