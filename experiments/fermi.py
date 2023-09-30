import matplotlib.pyplot as plt

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.fermi_configs import FermiGCE_MLP_CondFlowMatch as Configs

configs = Configs(dataset = '../data/fermi/fermi_data_galactic_coord.npy',
                  features = ['theta', 'phi', 'energy'],
                  preprocess = ['normalize', 'logit_transform', 'standardize'],
                  cuts = {'theta': [-20., 20.], 'phi': [4., 10.], 'energy': [1000, 2000]},
                  data_split_fracs = [0.6, 0.1, 0.3],
                  epochs = 1000,
                  early_stopping=20,
                  batch_size = 512,
                  warmup_epochs = 20,
                  lr = 1e-3,
                  dim_hidden = 128, 
                  device = 'cpu',
                  sigma = 1e-4,
                  solver='euler',
                  num_sampling_steps=500,
                  seed = 12345)

#...set working directory for results:

configs.set_workdir(path='../results', save_config=True)

#...define dataset :

from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 

dataset = FermiDataset(configs)

#...Train model:

from DynGenModels.models.deep_nets import MLP
from DynGenModels.dynamics.cnf.flowmatch import SimplifiedCondFlowMatching

net = MLP(configs)
cfm = DynGenModelTrainer(dynamics=SimplifiedCondFlowMatching(net, configs), 
                        dataloader=FermiDataLoader(dataset, configs),
                        configs=configs)

cfm.train()

#...sample from model

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             configs=configs, 
                             postprocessor=PostProcessFermiData,
                             solver='dopri5',
                             num_sampling_steps=100)


#...plots

bins=150 
fig, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].hist(x=pipeline.target[:,0], color='r', bins=bins, histtype='step', lw=0.75, density=True)
ax[0].hist(x=dataset.target[:,0], color='k', bins=bins, histtype='stepfilled', alpha=0.25, lw=0, density=True)

ax[1].hist(x=pipeline.target[:,1], color='r', bins=bins, histtype='step', lw=0.75, density=True)
ax[1].hist(x=dataset.target[:,1], color='k', bins=bins, histtype='stepfilled', alpha=0.25, lw=0, density=True)

ax[2].hist(x=pipeline.target[:,2], color='r', bins=bins, histtype='step', lw=0.75, density=True)
ax[2].hist(x=dataset.target[:,2], color='k', bins=bins, histtype='stepfilled', alpha=0.25, lw=0,density=True)

ax[0].set_xlabel(r'$\theta$')
ax[1].set_xlabel(r'$\phi$')
ax[2].set_xlabel(r'$E$')
plt.tight_layout()
plt.show()