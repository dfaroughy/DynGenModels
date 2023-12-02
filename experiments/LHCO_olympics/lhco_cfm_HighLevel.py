import torch
import numpy as np
import sys

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.lhco_configs import LHCOlympics_HighLevel_MLP_CondFlowMatch as Configs


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
                  DATA = 'LHCOlympicsHighLevel',
                  dataset = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode.h5', 
                  features = ['mjj', 'mj1', 'delta_m', 'tau21_1', 'tau21_2'],
                  cuts_sideband_low = {'mjj': [SB1_MIN, SB1_MAX]},  
                  cuts_sideband_high = {'mjj': [SB2_MIN, SB2_MAX]}, 
                  preprocess = ['standardize'],                            
                  num_dijets = NUM_DIJETS,  
                  dim_input = 5,
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
from DynGenModels.datamodules.lhco.dataprocess import PreProcessLHCOlympicsHighLevelData, PostProcessLHCOlympicsHighLevelData

pipeline = FlowMatchPipeline(trained_model=cfm, 
                             configs=configs, 
                             preprocessor=PreProcessLHCOlympicsHighLevelData,
                             postprocessor=PostProcessLHCOlympicsHighLevelData,
                             best_epoch_model=True)

pipeline.generate_samples(input_source=lhco.source)

#...plot results:

from utils import plot_interpolation

mjj_buffer = 100

plot_interpolation(lhco, pipeline, 
                    figsize=(18, 4.5),
                    mass_window=[SB1_MAX + mjj_buffer, SB2_MIN - mjj_buffer], 
                    bins=[(SB1_MIN-100, SB2_MAX+100, 40),  (0, 1200, 20), (-1250, 1250, 40), (0, 1.1, 0.02), (0, 1.1, 0.02)], 
                    log=False, 
                    density=True,
                    save_path=configs.workdir+'/interpolation.png')
