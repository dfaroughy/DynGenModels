import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from jetnet.datasets import JetNet

from DynGenModels.datamodules.jetnet.dataprocess import PreProcessJetNetData

class JetNetDataset(Dataset):

    def __init__(self, configs: dataclass):
        self.data_dir = configs.data_dir
        self.num_particles = configs.num_particles
        self.jet_types = configs.jet_types if isinstance(configs.jet_types, list) else [configs.jet_types]
        self.preprocess_methods = configs.preprocess 
        self.summary_stats = None
        
        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.particles_preprocess[idx]
        output['source'] = self.source[idx]
        output['mask'] = self.mask[idx]
        output['context'] = self.jets[idx]
        return output

    def __len__(self):
        return self.jets.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):

        args = {"jet_type": self.jet_types,  
                "data_dir": self.data_dir+ str(self.num_particles),
                "particle_features": ["etarel", "phirel", "ptrel", "mask"],
                "num_particles": self.num_particles,
                "jet_features": ["type", "pt", "eta", "mass"]}

        particle_data, jet_data = JetNet.getData(**args)
        self.particles = torch.Tensor(particle_data[..., :-1])
        self.mask = torch.Tensor(particle_data[..., -1])
        self.jets = torch.Tensor(jet_data)
        particles = PreProcessJetNetData(self.particles, methods=self.preprocess_methods)
        particles.preprocess()
        self.summary_stats = particles.summary_stats
        self.particles_preprocess = particles.features.clone()

    def get_source_data(self):
        self.source = torch.randn_like(self.particles, dtype=torch.float32)