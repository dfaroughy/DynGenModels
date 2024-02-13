import torch
from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.register_config_fermi import FermiGCE_MLP_FlowMatch as Configs

configs = Configs(# data params:
                  DATA = 'FermiGCE',
                  dataset = '../../data/fermi/fermi_data_galactic_coord.npy',
                  features = ['theta', 'phi', 'energy'],
                  preprocess = ['normalize'],
                  cuts = {'theta': [-10., 10.], 'phi': [-5., 10.], 'energy': [1000, 2000]},
                  data_split_fracs = [0.75, 0.25, 0.0],
                  # training params:
                  DEVICE = 'cuda:0',
                  EPOCHS = 1000,
                  batch_size = 16000,
                  print_epochs = 20,
                  early_stopping = 200,
                  min_epochs = 1000,
                  lr = 1e-3,
                  optimizer = 'Adam',
                  fix_seed = 10,
                  # model params:
                  DYNAMICS = 'FlowMatch',
                  MODEL = 'MLP',
                  dim_hidden = 256,
                  num_layers = 5,
                  sigma = 0.0,
                  t0 = 0.0,
                  t1 = 1.0,
                  # sampling params:
                  solver = 'rk4',
                  num_sampling_steps = 1000,
                  num_gen_samples = 100000
                )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset and model:

from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 
from DynGenModels.models.deep_nets import MLP
from DynGenModels.dynamics.cnf.flowmatch import FlowMatching

# print(MLP(configs))

fermi = FermiDataset(configs)
fm = DynGenModelTrainer(dynamics = FlowMatching(configs),
                         model = MLP(configs), 
                         dataloader = FermiDataLoader(fermi, configs), 
                         configs = configs)

#...train model:

fm.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData 

pipeline = FlowMatchPipeline(trained_model=fm, 
                             source_input=torch.randn(configs.num_gen_samples, configs.dim_input),
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
              bins=200, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'],
              num_particles=configs.num_gen_samples)

results_2D_plots(pipeline.target,
                 save_path=configs.workdir + '/fermi_features_2D.pdf',  
                 gridsize=200)
