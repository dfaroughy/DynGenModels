import torch
import numpy as np
import sys

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.lhco_configs import LHCOlympics_LowLevel_MLP_CondFlowMatch as Configs

CUDA = 'cuda:{}'.format(sys.argv[1]) if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = int(sys.argv[2])
LR = float(sys.argv[3])
DIM_HIDDEN = int(sys.argv[4])
DIM_TIME_EMB = int(sys.argv[5])
NUM_LAYERS = int(sys.argv[6])
EXCHANGE_TARGET_WITH_SOURCE = bool(int(sys.argv[7]))
DYNAMICS = sys.argv[8]
SIGMA = float(sys.argv[9])
SB1_MIN = float(sys.argv[10])
SB1_MAX = float(sys.argv[11])
SB2_MIN = float(sys.argv[12])
SB2_MAX = float(sys.argv[13])
NUM_DIJETS = int(sys.argv[14])

configs = Configs(# data:
                  DATA = 'LHCOlympicsLowLevel',
                  dataset = '../../data/LHCOlympics2020/events_anomalydetection_low_level_ptepm.h5', 
                  # features = ['px_j1', 'py_j1', 'pz_j1', 'e_j1', 'px_j2', 'py_j2', 'pz_j2', 'e_j2'],
                  features = ['pt_j1', 'eta_j1', 'phi_j1', 'm_j1', 'pt_j1', 'eta_j2', 'phi_j2', 'm_j2'],
                  cuts_sideband_low = {'mjj': [SB1_MIN, SB1_MAX]},  
                  cuts_sideband_high = {'mjj': [SB2_MIN, SB2_MAX]}, 
                  preprocess = ['normalize', 'logit_transform', 'standardize'],                            
                  num_dijets = NUM_DIJETS,  
                  dim_input = 8,
                  # training params:   
                  DEVICE = CUDA,
                  EPOCHS = 5000,
                  batch_size = BATCH_SIZE,
                  print_epochs = 10,
                  early_stopping = 75,
                  min_epochs = 500,
                  data_split_fracs = [0.85, 0.15, 0.0],
                  lr = LR,
                  optimizer = 'Adam',
                  fix_seed = 12345,
                  # model params:
                  DYNAMICS = DYNAMICS,
                  MODEL = 'MLP_backward' if EXCHANGE_TARGET_WITH_SOURCE else 'MLP_forward',
                  dim_hidden = DIM_HIDDEN,
                  dim_time_emb = DIM_TIME_EMB,
                  num_layers = NUM_LAYERS,
                  activation = 'ReLU',
                  sigma = SIGMA,
                  t0 = 0.0,
                  t1 = 1.0,
                  # sampling params:
                  solver = 'midpoint',
                  num_sampling_steps = 1000
                )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset and model:

from DynGenModels.datamodules.lhco.datasets import LHCOlympicsLowLevelDataset 
from DynGenModels.datamodules.lhco.dataloader import LHCOlympicsDataLoader 
from DynGenModels.models.deep_nets import MLP

if DYNAMICS == 'OptimalTransportFlowMatching':
  from DynGenModels.dynamics.cnf.condflowmatch import OptimalTransportFlowMatching as dynamics

if DYNAMICS == 'SchrodingerBridgeFlowMatching':
  from DynGenModels.dynamics.cnf.condflowmatch import SchrodingerBridgeFlowMatching as dynamics

lhco = LHCOlympicsLowLevelDataset(configs, exchange_target_with_source=EXCHANGE_TARGET_WITH_SOURCE)
cfm = DynGenModelTrainer(dynamics = dynamics(configs),
                         model = MLP(configs), 
                         dataloader = LHCOlympicsDataLoader(lhco, configs), 
                         configs = configs)

#...train model:

cfm.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import FlowMatchPipeline 
from DynGenModels.datamodules.lhco.dataprocess import PreProcessLHCOlympicsLowLevelData, PostProcessLHCOlympicsLowLevelData

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             configs=configs, 
                             preprocessor=PreProcessLHCOlympicsLowLevelData,
                             postprocessor=PostProcessLHCOlympicsLowLevelData,
                             best_epoch_model=True)

pipeline.generate_samples(input_source=lhco.source)

#...plot results:

from utils import plot_interpolation_low_level

mjj_buffer = 50.0

plot_interpolation_low_level(lhco, pipeline, 
                             time_stop_feature='mjj',
                             coords="pt_eta_phi_m",
                             features=['mjj', 'pt_j1', 'eta_j1', 'phi_j1', 'm_j1'],
                             bins=[(SB1_MIN-mjj_buffer, SB2_MAX+mjj_buffer, 15), (1000, 2500, 20), (-2.5, 2.5, 0.075), (-2.5, 2.5, 0.075), (0, 1300, 20)], 
                             figsize=(22, 4.5),
                             mass_window=[SB1_MAX + mjj_buffer, SB2_MIN - mjj_buffer],
                             log=False, 
                             density=True,
                             save_path=configs.workdir+'/interpolation_low_level_j1.png')

plot_interpolation_low_level(lhco, pipeline, 
                             time_stop_feature='mjj',
                             coords="pt_eta_phi_m",
                             features=['mjj', 'pt_j2', 'eta_j2', 'phi_j2', 'm_j2'],
                             bins=[(SB1_MIN-mjj_buffer, SB2_MAX+mjj_buffer, 15), (500, 2000, 20), (-2.5, 2.5, 0.075), (-2.5, 2.5, 0.075), (0, 1300, 20)],
                             figsize=(22, 4.5),
                             mass_window=[SB1_MAX + mjj_buffer, SB2_MIN - mjj_buffer],
                             log=False, 
                             density=True,
                             save_path=configs.workdir+'/interpolation_low_level_j2.png')

plot_interpolation_low_level(lhco, pipeline, 
                             coords="pt_eta_phi_m",
                             time_stop_feature='mjj',
                             features=['mjj', 'delta_Rjj', 'delta_mjj', 'delta_ptjj', 'delta_etajj'],
                             bins=[(SB1_MIN-mjj_buffer, SB2_MAX+mjj_buffer, 15), (2.5, 4.5, 0.02), (0, 1000, 10), (0, 1000, 10), (0, 3, 0.04)], 
                             figsize=(22, 4.5),
                             mass_window=[configs.cuts_sideband_low['mjj'][1] + mjj_buffer, configs.cuts_sideband_high['mjj'][0] - mjj_buffer], 
                             log=False, 
                             density=True,
                             save_path=configs.workdir+'/interpolation_high_level_dijet.png',
                             show=True)