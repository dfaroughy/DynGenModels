
from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.fermi_configs import FermiGCE_MAF_RQS_NormFlow as Configs

configs = Configs(dataset = '../../data/fermi/fermi_data_galactic_coord.npy',
                features = ['theta', 'phi', 'energy'],
                preprocess = ['normalize', 'logit_transform', 'standardize'],
                cuts = {'theta': [-10., 10.], 'phi': [-5., 10.], 'energy': [1000, 2000]},
                data_split_fracs = [0.8, 0.2, 0.0],
                EPOCHS = 10000,
                batch_size = 15000,
                print_epochs = 20,
                early_stopping = 100,
                min_epochs = 1000,
                lr = 1e-4,
                dim_hidden = 256, 
                dropout = 0.1,
                DEVICE = 'cuda:0',
                optimizer = 'Adam',
                seed = 12345,
                tail_bound = 10, 
                num_transforms = 10,
                num_blocks = 3,
                num_bins = 30,
                num_gen_samples = 1000000
                )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset :

from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 
from DynGenModels.models.nflow_nets import MAFPiecewiseRQS
from DynGenModels.dynamics.nf.normflows import NormalizingFlow

fermi = FermiDataset(configs)
dataloader = FermiDataLoader(fermi, configs)
net = MAFPiecewiseRQS(configs)
dynamics = NormalizingFlow(net, configs)
maf = DynGenModelTrainer(dynamics=dynamics, dataloader=dataloader, configs=configs)

#...train model:

maf.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import NormFlowPipeline
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData 

pipeline = NormFlowPipeline(trained_model=maf, 
                            configs=configs, 
                            postprocessor=PostProcessFermiData,
                            best_epoch_model=False)

pipeline_best = NormFlowPipeline(trained_model=maf, 
                                 configs=configs, 
                                 postprocessor=PostProcessFermiData,
                                 best_epoch_model=True)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def results_plots(data, generated=None, save_dir=None, features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'], num_particles=None):
    num_particles = 100000000 if num_particles is None else num_particles
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[5, 1])
    gs.update(hspace=0.1) 
    
    for idx, feature in enumerate(features):
        ax = fig.add_subplot(gs[idx])
        h1, bins, _ = ax.hist(data[..., idx].flatten()[:num_particles], bins=100, color='silver', density=True)
        if generated is not None:
            h2, _, _ = ax.hist(generated[..., idx].flatten()[:num_particles], bins=100, color=['gold', 'darkblue', 'darkred'][idx], histtype='step', density=True, lw=0.75)
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
              tick.label.set_fontsize(5)  
            if idx == 0:
                ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_yticks([0.5, 1, 1.5])
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()

#...plot results:

results_plots(fermi.target, generated=pipeline.target, save_dir=configs.workdir + '/fermi_features.pdf', features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])
results_plots(fermi.target, generated=pipeline_best.target, save_dir=configs.workdir + '/fermi_features_best.pdf', features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])

#...plot 2D projections:

coord = [r'$\theta$', r'$\phi$', r'$E$']
color=['gold', 'darkblue', 'darkred']
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for i, j in [(0,1), (1,2), (2,0)]:
    ax[i].hexbin(pipeline.target[..., i], pipeline.target[..., j], cmap='plasma', gridsize=200)
    ax[i].set_xlabel(coord[i])
    ax[i].set_ylabel(coord[j])
plt.tight_layout()
plt.savefig(configs.workdir + '/fermi_features_2D.pdf')
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for i, j in [(0,1), (1,2), (2,0)]:
    ax[i].hexbin(pipeline_best.target[..., i], pipeline_best.target[..., j], cmap='plasma', gridsize=200)
    ax[i].set_xlabel(coord[i])
    ax[i].set_ylabel(coord[j])
plt.tight_layout()
plt.savefig(configs.workdir + '/fermi_features_2D_best.pdf')
plt.show()