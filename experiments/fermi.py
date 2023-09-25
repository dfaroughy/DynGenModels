import torch
import matplotlib.pyplot as plt
import seaborn as sns

from DynGenModels.datamodules.fermi.datasets import FermiDataset
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader

from DynGenModels.models.resnet import ResNet
from DynGenModels.models.resnet_config import ResNetConfig as Config
from DynGenModels.trainer.trainer import FlowMatchTrainer
from DynGenModels.dynamics.flowmatch import SimplifiedCondFlowMatching
from DynGenModels.pipelines.FlowMatchPipeline import FlowMatchPipeline

config = Config(dataset = 'fermi_data_galactic_coord.npy',
                features    = ['theta', 'phi', 'energy'],
                preprocess  = ['normalize', 'logit_transform', 'standardize'],
                cuts = {'theta': [-10., 10.], 'phi': [4., 10.], 'energy': [1000, 2000]},
                data_split_fracs = [0.6, 0.1, 0.3],
                epochs = 1000,
                batch_size = 2048,
                early_stopping = 10,
                warmup_epochs = 0,
                dim_hidden = 128, 
                num_layers = 3,
                device = 'cpu',
                solver = 'dopri5',
                num_sampling_steps = 100
                )

root_dir =  '/home/df630/' if 'cuda' in config.device else '/Users/dario/Dropbox/PROJECTS/ML/'
root_dir += 'DynGenModels'

if __name__ == "__main__":

    resnet = ResNet(model_config=config)
    config.set_workdir(root_dir + '/results', save_config=True)

    datasets = FermiDataset(dir_path = root_dir + '/data/fermi', 
                            dataset = config.dataset,
                            cuts = config.cuts,
                            preprocess = config.preprocess,
                            ) 
    dataloader = FermiDataLoader(datasets=datasets, data_split_fracs=config.data_split_fracs, batch_size=config.batch_size)

    CFM = FlowMatchTrainer(dynamics = SimplifiedCondFlowMatching(resnet), 
                           dataloader = dataloader,
                           workdir = config.workdir,
                           lr = config.lr,
                           epochs = config.epochs,
                           early_stopping = config.early_stopping,
                           warmup_epochs = config.warmup_epochs)
    
    CFM.train()

    pipeline = FlowMatchPipeline(trained_model=CFM, 
                                 postprocessor=PostProcessFermiData,
                                 solver=config.solver, 
                                 num_sampling_steps=config.num_sampling_steps)
    

    sns.histplot(x=pipeline.target[:,0], color='k', bins=250,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()

    sns.histplot(x=pipeline.target[:,1], color='k', bins=250,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()

    sns.histplot(x=pipeline.target[:,2], color='k', bins=250,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()