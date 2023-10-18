import torch
from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.fermi_configs import FermiGCE_MLP_CondFlowMatch as Configs

configs = Configs(# data params:
                  DATA = 'FermiGCE',
                  dataset = '../../data/fermi/fermi_data_galactic_coord.npy',
                  features = ['theta', 'phi', 'energy'],
                  preprocess = ['normalize', 'logit_transform', 'standardize'],
                  cuts = {'theta': [-10., 10.], 'phi': [-5., 10.], 'energy': [1000, 2000]},
                  data_split_fracs = [0.8, 0.2, 0.0],
                  # training params:
                  DEVICE = 'cpu',
                  EPOCHS = 100,
                  batch_size = 15000,
                  print_epochs = 20,
                  early_stopping = 100,
                  min_epochs = 1000,
                  lr = 1e-4,
                  optimizer = 'Adam',
                  fix_seed = 12345,
                  # model params:
                  DYNAMICS = 'CondFlowMatch',
                  MODEL = 'MLP',
                  dim_hidden = 128,
                  num_layers = 4,
                  sigma = 1e-5,
                  t0 = 0.0,
                  t1 = 1.0,
                  # sampling params:
                  solver = 'midpoint',
                  num_sampling_steps = 100
                )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset and model:

from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 
from DynGenModels.models.deep_nets import MLP
from DynGenModels.dynamics.cnf.condflowmatch import SimplifiedCondFlowMatching

fermi = FermiDataset(configs)
cfm = DynGenModelTrainer(dynamics = SimplifiedCondFlowMatching(configs),
                         model = MLP(configs), 
                         dataloader = FermiDataLoader(fermi, configs), 
                         configs = configs)

#...train model:

cfm.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData 

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             source_input=torch.randn(fermi.target.shape[0], configs.dim_input),
                             configs=configs, 
                             postprocessor=PostProcessFermiData, 
                             best_epoch_model=True)

#...plot results:

from utils import results_plots, results_2D_plots

results_plots(data=fermi.target, 
              generated=pipeline.target, 
              model = 'CFM', 
              comparator='pull',
              save_path=configs.workdir + '/fermi_features.pdf', 
              bins=300, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'],
              num_particles=pipeline.target.shape[0])

# results_2D_plots(pipeline.target,
#                  save_path=configs.workdir + '/fermi_features_2D.pdf',  
#                  gridsize=200)
