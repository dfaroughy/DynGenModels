import torch

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.fermi_configs import FermiGCE_ResNet_CondFlowMatch as Configs

configs = Configs(DATA ='FermiGCE',
                  dataset = '../../data/fermi/fermi_data_galactic_coord.npy',
                  features = ['theta', 'phi', 'energy'],
                  preprocess = ['normalize', 'logit_transform', 'standardize'],
                  cuts = {'theta': [-10., 10.], 'phi': [-5., 10.], 'energy': [1000, 2000]},
                  data_split_fracs = [0.8, 0.2, 0.0],
                  EPOCHS = 10000,
                  batch_size = 15000,
                  print_epochs = 10,
                  early_stopping = 120,
                  min_epochs = 100,
                  lr = 1e-4,
                  dim_hidden = 256, 
                  DEVICE = 'cuda:1',
                  optimizer = 'Adam',
                  fix_seed = 12345,
                  num_blocks = 10,
                  num_block_layers = 3,
                  sigma = 0.0,
                  solver = 'midpoint',
                  num_sampling_steps = 500
                  )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset and model:

from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 
from DynGenModels.models.deep_nets import ResNet
from DynGenModels.dynamics.cnf.condflowmatch import SimplifiedCondFlowMatching

fermi = FermiDataset(configs)
dataloader = FermiDataLoader(fermi, configs)
net = ResNet(configs)
dynamics = SimplifiedCondFlowMatching(net, configs)
cfm = DynGenModelTrainer(dynamics=dynamics, dataloader=dataloader, configs=configs)

#...train model:

cfm.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData 

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             source_input=torch.randn(fermi.target.shape[0], configs.dim_input),
                             configs=configs, 
                             postprocessor=PostProcessFermiData,
                             best_epoch_model=False)

pipeline_best = FlowMatchPipeline(trained_model=cfm, 
                             source_input=torch.randn(fermi.target.shape[0], configs.dim_input),
                             configs=configs, 
                             postprocessor=PostProcessFermiData,
                             best_epoch_model=True)


#...plot results:

from utils import results_plots, results_2D_plots

results_plots(data=fermi.target, 
              generated=pipeline.target, 
              comparator='pull',
              model = 'Flow-Matching', 
              save_path=configs.workdir + '/fermi_features.pdf', 
              bins=300, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])

results_plots(data=fermi.target, 
              generated=pipeline_best.target, 
              comparator='pull',
              model = 'Flow-Matching', 
              save_path=configs.workdir + '/fermi_features_best.pdf', 
              bins=300, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])

results_2D_plots(pipeline.target,
                 save_path=configs.workdir + '/fermi_features_2D.pdf',  
                 gridsize=200)

results_2D_plots(pipeline_best.target,
                 save_path=configs.workdir + '/fermi_features_2D_best.pdf',  
                 gridsize=200)