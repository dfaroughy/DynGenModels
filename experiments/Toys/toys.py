import torch
import numpy as np
import sys


from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.configs.toys_configs import Gauss_2_Moons_MLP_FlowMatch as Configs



CUDA = 'cuda:{}'.format(sys.argv[1])
EXCHANGE_TARGET_WITH_SOURCE = bool(int(sys.argv[2]))
EPOCHS = int(sys.argv[3])
DYNAMICS = sys.argv[4]

configs = Configs(DATA = '8Gauss2Moons',
                num_points = 10000,
                data_split_fracs = [0.8976, 0.0, 0.1024],
                batch_size = 256,
                EPOCHS = 1000,
                print_epochs = 10,
                DEVICE= CUDA,
                fix_seed = 12345,
                lr = 1e-3,
                DYNAMICS = DYNAMICS,
                MODEL = 'MLP_bwd' if EXCHANGE_TARGET_WITH_SOURCE else 'MLP_fwd',
                dim_hidden = 64, 
                sigma = 0.1,
                solver='midpoint',
                num_sampling_steps=200  
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