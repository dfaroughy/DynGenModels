import torch
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.jetnet_configs import JetNet_EPiC_CondFlowMatch as Configs

configs = Configs(data_dir = '../../data/jetnet',
                  features = ['eta_rel', 'phi_rel', 'pt_rel'],
                  preprocess=['standardize'],
                  num_particles = 30,
                  cuts = {'num_constituents': 30},
                  jet_types = 't',
                  data_split_fracs = [0.8, 0.2, 0.0],
                  epochs = 5,
                  batch_size = 1024,
                  lr = 1e-3,
                  gradient_clip = 1.0,
                  pooling = 'attention',
                  dim_global = 10,
                  dim_hidden = 128, 
                  num_epic_layers = 6,
                  sigma = 1e-5,
                  solver='midpoint',
                  num_sampling_steps=500,
                  device='cuda:1')

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define setup and train model :

from DynGenModels.datamodules.jetnet.datasets import JetNetDataset 
from DynGenModels.datamodules.jetnet.dataloader import JetNetDataLoader 
from DynGenModels.models.deep_sets import EPiC
from DynGenModels.dynamics.cnf.condflowmatch import SimplifiedCondFlowMatching

tops = JetNetDataset(configs)
dataloader = JetNetDataLoader(tops, configs)
net = EPiC(configs)
dynamics = SimplifiedCondFlowMatching(net, configs)
cfm = DynGenModelTrainer(dynamics=dynamics, dataloader=dataloader, configs=configs)
cfm.train()

#...define pipeline and generate data:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
from DynGenModels.datamodules.jetnet.dataprocess import PostProcessJetNetData 

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             source_input=torch.randn(20000, 30, 3),
                             configs=configs, 
                             postprocessor=PostProcessJetNetData)

#...plot generated data:

def results_plots(jetnet_data, generated=None, save_dir=None, features=[r'$\Delta\eta$', r'$\Delta\phi$', r'$p^{\rm rel}_T$'], num_particles=100000):
    fig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 1])
    gs.update(hspace=0.05) 
    
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[idx])
        h1, bins, _ = ax.hist(jetnet_data[..., idx].flatten()[:num_particles], bins=100, log=True, color='silver', density=True)
        if generated is not None:
            h2, _, _ = ax.hist(generated[..., idx].flatten()[:num_particles], bins=100, log=True, color=['gold', 'darkblue', 'darkred'][idx], histtype='step', density=True, lw=0.75)
            ax.set_xticklabels([])
            ax.set_xticks([])
            for tick in ax.yaxis.get_major_ticks():
               tick.label.set_fontsize(8)
        else:
            ax.set_xlabel(feature)
        
        # Ratio plot
        if generated is not None:
            ax_ratio = fig.add_subplot(gs[idx + 3])
            ratio = np.divide(h1, h2, out=np.ones_like(h2), where=h2 != 0)
            ax_ratio.plot(0.5 * (bins[:-1] + bins[1:]), ratio, color=['gold', 'darkblue', 'darkred'][idx],lw=0.75)
            ax_ratio.set_ylim(0.5, 1.5, 0) # Adjust this as needed
            ax_ratio.set_xlabel(feature)
            ax_ratio.axhline(1, color='gray', linestyle='--', lw=0.75)
            for tick in ax_ratio.xaxis.get_major_ticks():
               tick.label.set_fontsize(7)
            for tick in ax_ratio.yaxis.get_major_ticks():
              tick.label.set_fontsize(8)  
            if idx == 0:
                ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_yticks([0.5, 1, 1.5])
    if save_dir is not None:
        plt.savefig(save_dir + '/particle_features.pdf')
    plt.show()


results_plots(tops.particles, pipeline.target, save_dir=configs.workdir)