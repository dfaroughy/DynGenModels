import torch

from DynGenModels.datamodules.fermi.datasets import FermiDataset
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader
from DynGenModels.models.resnet import ResNet
from DynGenModels.models.resnet_config import ResNetConfig as Config
from DynGenModels.trainer.trainer import FlowMatchTrainer
from DynGenModels.dynamics.flowmatch import SimplifiedCondFlowMatching
from DynGenModels.pipelines.FlowMatchPipeline import FlowMatchPipeline

config = Config(dataset = 'fermi_data_galactic_coord.npy',
                features    = ['theta', 'phi', 'energy'],
                preprocess  = ['standardize'],
                cuts = {'theta': [-10., 10.], 'phi': [4., 10.], 'energy': [1000, 2000]},
                data_split_fracs = [0.6, 0.1, 0.3],
                epochs = 30,
                batch_size = 128,
                warmup_epochs= 5,
                dim_hidden = 128, 
                num_layers = 3,
                device = 'cpu'
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

    CFM = FlowMatchTrainer(dynamics = SimplifiedCondFlowMatching(model=resnet), 
                           dataloader = dataloader,
                           workdir = config.workdir,
                           lr = config.lr,
                           early_stopping = config.epochs,
                           warmup_epochs = config.warmup_epochs)
    
    CFM.train()
    source = torch.rand((30000, config.dim_input))
    pipeline = FlowMatchPipeline(pretrained_model=CFM, source_data=source, solver='dopri5', sampling_steps=100)
    target = pipeline.target

    print(pipeline.trajectories.shape, target.shape)
    print(target)

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.histplot(x=pipeline.trajectories[0][:,0], color='k', bins=120,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    sns.histplot(x=pipeline.trajectories[100][:,0], color='purple', bins=120,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    sns.histplot(x=pipeline.trajectories[200][:,0], color='b', bins=120,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    sns.histplot(x=pipeline.trajectories[300][:,0], color='g', bins=120,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    sns.histplot(x=pipeline.trajectories[-1][:,0], color='r', bins=120,log_scale=(False, True), element="step", lw=0.75, fill=False, alpha=1) 
    plt.show()