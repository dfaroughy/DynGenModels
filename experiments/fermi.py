import torch
import matplotlib.pyplot as plt
import seaborn as sns

from DynGenModels.datamodules.fermi.datasets import FermiDataset
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData

from DynGenModels.trainer.trainer import FlowMatchTrainer
from DynGenModels.pipelines.FlowMatchPipeline import FlowMatchPipeline

from DynGenModels.dynamics.flowmatch import SimplifiedCondFlowMatching
from DynGenModels.configs.fermi_configs import FermiMLPConfig
from DynGenModels.models.mlp import MLP


config = FermiMLPConfig(dataset = '../data/fermi/fermi_data_galactic_coord.npy',
                        features    = ['theta', 'phi', 'energy'],
                        preprocess  = ['normalize', 'logit_transform', 'standardize'],
                        cuts = {'theta': [-10., 10.], 'phi': [4., 10.], 'energy': [1000, 2000]},
                        data_split_fracs = [0.6, 0.1, 0.3],
                        epochs = 10,
                        batch_size = 512,
                        early_stopping = 10,
                        warmup_epochs = 0,
                        dim_hidden = 512, 
                        device = 'cpu',
                        solver = 'dopri5',
                        num_sampling_steps = 100
                        )

root_dir =  '/home/df630/' if 'cuda' in config.device else '/Users/dario/Dropbox/PROJECTS/ML/'
root_dir += 'DynGenModels'

if __name__ == "__main__":

    mlp = MLP(config)
    config.set_workdir(root_dir + '/results', save_config=True)
    datasets = FermiDataset(config) 
    dataloader = FermiDataLoader(datasets, config)

    #...define and train model:

    CFM = FlowMatchTrainer(dynamics=SimplifiedCondFlowMatching(mlp), 
                           dataloader=dataloader,
                           config=config)
    
    CFM.train()
    
    #...sampling pipeline:

    pipeline = FlowMatchPipeline(trained_model=CFM, 
                                 postprocessor=PostProcessFermiData,
                                 config=config)
    
    sns.histplot(x=pipeline.target[:,0], color='k', bins=250,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()

    sns.histplot(x=pipeline.target[:,1], color='k', bins=250,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()

    sns.histplot(x=pipeline.target[:,2], color='k', bins=250,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()