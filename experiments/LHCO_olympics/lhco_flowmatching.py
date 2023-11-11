import torch
import numpy as np
import sys

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.lhco_configs import LHCOlympics_MLP_CondFlowMatch as Configs

BATCH_SIZE = int(sys.argv[1])
LR = float(sys.argv[2])
DIM_HIDDEN = int(sys.argv[3])
CUDA = 'cuda:{}'.format(sys.argv[4])

configs = Configs(# data:
                  DATA = 'LHCOlympics',
                  dataset = '../../data/LHCOlympics2020/events_anomalydetection_dijets.h5',
                  cuts_sideband_low = {'mjj': [2700, 3100]},  
                  cuts_sideband_high = {'mjj': [3900, 13000]}, 
                  dim_input = 8,
                  preprocess = None,                            
                  num_dijets = 60000,  
                  # training params:   
                  DEVICE = CUDA,
                  EPOCHS = 10000,
                  batch_size = BATCH_SIZE,
                  print_epochs = 500,
                  early_stopping = None,
                  min_epochs = 5000,
                  data_split_fracs = [0.8, 0.2, 0.0],
                  lr = LR,
                  optimizer = 'Adam',
                  fix_seed = 12345,
                  # model params:
                  MODEL = 'MLP',
                  dim_hidden = DIM_HIDDEN,
                  num_layers = 3,
                  sigma = 1e-5,
                  t0 = 0.0,
                  t1 = 1.0,
                  # sampling params:
                  solver = 'midpoint',
                  num_sampling_steps = 1000
                )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset and model:

from DynGenModels.datamodules.lhco.datasets import LHCOlympicsDataset 
from DynGenModels.datamodules.lhco.dataloader import LHCOlympicsDataLoader 
from DynGenModels.models.deep_nets import MLP
from DynGenModels.dynamics.cnf.condflowmatch import OptimalTransportFlowMatching

lhco = LHCOlympicsDataset(configs)

cfm = DynGenModelTrainer(dynamics = OptimalTransportFlowMatching(configs),
                         model = MLP(configs), 
                         dataloader = LHCOlympicsDataLoader(lhco, configs), 
                         configs = configs)

#...train model:

cfm.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             source_input=lhco.source,
                             configs=configs, 
                             best_epoch_model=True)

#...plot results:

from utils import plot_jet_features, plot_dijet_mass

STEP = configs.num_sampling_steps/2
D_STEP  = configs.num_sampling_steps/4

plot_jet_features(lhco, pipeline.trajectories, 'p_t', xlim=(800, 2500, 25), time_step=STEP, d_step=D_STEP, save_path=configs.workdir + '/jet_pt.pdf')
plot_jet_features(lhco, pipeline.trajectories, '\eta', xlim=(-2, 2, 0.05), time_step=STEP, d_step=D_STEP, save_path=configs.workdir + '/jet_eta.pdf')
plot_jet_features(lhco, pipeline.trajectories, '\phi', xlim=(-1, 12, 0.1),  time_step=STEP, d_step=D_STEP, save_path=configs.workdir + '/jet_phi.pdf')
plot_jet_features(lhco, pipeline.trajectories, 'm', xlim=(0, 1000, 10), time_step=STEP, d_step=D_STEP, save_path=configs.workdir + '/jet_mass.pdf')
plot_dijet_mass(lhco, pipeline.trajectories, time_step=STEP, d_step=D_STEP,  bins=np.arange(2700, 6000, 20), save_path=configs.workdir + '/dijet_inv_mass.pdf')
