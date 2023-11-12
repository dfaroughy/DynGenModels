import torch
import numpy as np
import sys

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.lhco_configs import LHCOlympics_HighLevel_MLP_CondFlowMatch as Configs

BATCH_SIZE = int(sys.argv[1])
LR = float(sys.argv[2])
DIM_HIDDEN = int(sys.argv[3])
CUDA = 'cuda:{}'.format(sys.argv[4])
EXCHANGE_TARGET_WITH_SOURCE = bool(int(sys.argv[5]))
DYNAMICS = sys.argv[6]

configs = Configs(# data:
                  DATA = 'LHCOlympics',
                  dataset = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode.h5', 
                  features = ['mjj', 'mj1', 'delta_m', 'tau21_1', 'tau21_2'],
                  cuts_sideband_low = {'mjj': [2600, 3200]},  
                  cuts_sideband_high = {'mjj': [3800, 4400]}, 
                  preprocess = None,                            
                  num_dijets = 75166,  
                  # training params:   
                  DEVICE = CUDA,
                  EPOCHS = 1000,
                  batch_size = BATCH_SIZE,
                  print_epochs = 20,
                  early_stopping = 50,
                  min_epochs = 500,
                  data_split_fracs = [0.85, 0.15, 0.0],
                  lr = LR,
                  optimizer = 'Adam',
                  fix_seed = 12345,
                  # model params:
                  DYNAMICS = DYNAMICS,
                  MODEL = 'MLP_backward' if EXCHANGE_TARGET_WITH_SOURCE else 'MLP_forward',
                  dim_hidden = DIM_HIDDEN,
                  num_layers = 4,
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

from DynGenModels.datamodules.lhco.datasets import LHCOlympicsHighLevelDataset 
from DynGenModels.datamodules.lhco.dataloader import LHCOlympicsDataLoader 
from DynGenModels.models.deep_nets import MLP

if DYNAMICS == 'OptimalTransportFlowMatching':
  from DynGenModels.dynamics.cnf.condflowmatch import OptimalTransportFlowMatching as dynamics
if DYNAMICS == 'SchrodingerBridgeFlowMatching':
  from DynGenModels.dynamics.cnf.condflowmatch import SchrodingerBridgeFlowMatching as dynamics

lhco = LHCOlympicsHighLevelDataset(configs, exchange_target_with_source=EXCHANGE_TARGET_WITH_SOURCE)
cfm = DynGenModelTrainer(dynamics = dynamics(configs),
                         model = MLP(configs), 
                         dataloader = LHCOlympicsDataLoader(lhco, configs), 
                         configs = configs)

#...train model:

cfm.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             configs=configs, 
                             best_epoch_model=True)

pipeline.generate_samples(input_source=lhco.source)

#...plot results:

from utils import plot_interpolation

plot_interpolation(lhco, pipeline, figsize=(18, 4.5),
                    mass_window=[configs.cuts_sideband_low['mjj'][1], configs.cuts_sideband_high['mjj'][0]], 
                    bins=[(2500, 5000, 40), (0, 1400, 50), (-1250, 1250, 100), (-0.25, 1.25, 0.05), (-0.25, 1.25, 0.05)], 
                    log=False, 
                    density=True,
                    save_path=configs.workdir+'/interpolation.png')