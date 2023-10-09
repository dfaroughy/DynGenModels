import torch
import numpy as np
import matplotlib.pyplot as plt 

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.fermi_configs import FermiGCE_Couplings_RQS_NormFlow as Configs

configs = Configs(dataset = '../../data/fermi/fermi_data_galactic_coord.npy',
                  features = ['theta', 'phi', 'energy'],
                  preprocess =['normalize', 'logit_transform', 'standardize'], 
                  cuts = {'theta': [-10., 10.], 'phi': [4., 10.], 'energy': [1000, 2000]},
                  data_split_fracs = [0.8, 0.2, 0.0],
                  epochs = 10000,
                  warmup_epochs = 100,
                  early_stopping = 100,
                  print_epochs = 100, 
                  batch_size = 2048,
                  lr = 1e-3,
                  num_transforms = 6,
                  num_blocks = 3,
                  mask = 'checkerboard',
                  dim_hidden = 256, 
                  use_batch_norm = True,
                  num_bins = 20,
                  tail_bound = 10, 
                  num_gen_samples = 100000,
                  device = 'cuda:0')

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset and model:

from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 
from DynGenModels.models.nflow_nets import CouplingsPiecewiseRQS 
from DynGenModels.dynamics.nf.normflows import NormalizingFlow

fermi = FermiDataset(configs)
dataloader = FermiDataLoader(fermi, configs)
net = CouplingsPiecewiseRQS(configs)
dynamics = NormalizingFlow(net, configs)
nf = DynGenModelTrainer(dynamics=dynamics, dataloader=dataloader, configs=configs)

#...train model:

nf.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import NormFlowPipeline
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData 

pipeline = NormFlowPipeline(trained_model=nf, 
                            configs=configs, 
                            postprocessor=PostProcessFermiData,
                            best_epoch_model=False)

pipeline_best = NormFlowPipeline(trained_model=nf, 
                                 configs=configs, 
                                 postprocessor=PostProcessFermiData,
                                 best_epoch_model=True)

coord = [r'$\theta$', r'$\phi$', r'$E$']
color=['gold', 'darkblue', 'darkred']

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for i in range(3):
    ax[i].hist(fermi.target[..., i], bins=100, color='silver', density=True)
    ax[i].hist(pipeline.target[..., i], bins=100, color=color[i], histtype='step', density=True)
    ax[i].hist(pipeline_best.target[..., i], bins=100, color=color[i], histtype='step', ls=':',density=True)
    ax[i].set_xlabel(coord[i])
plt.tight_layout()
plt.savefig(configs.workdir + '/fermi_coords.pdf')

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for i, j in [(0,1), (1,2), (2,0)]:
    ax[i].hexbin(pipeline.target[..., i], pipeline.target[..., j], cmap='plasma', gridsize=200)
    ax[i].set_xlabel(coord[i])
    ax[i].set_ylabel(coord[j])
plt.tight_layout()
plt.show()
plt.savefig(configs.workdir + '/fermi_coords_2D.pdf')
