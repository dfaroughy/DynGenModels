from DynGenModels.configs.registered_experiments import Config_JetClass_EPiC_CondFlowMatch, Config_JetClass_DeepSets_CondFlowMatch
from DynGenModels.models.experiment import Experiment

cfm = Experiment(Config_JetClass_EPiC_CondFlowMatch,
                 NAME = 'qcd_to_top_gauss',
                 DATA_SOURCE = 'qcd',
                 DATA_TARGET = 'top',
                 DYNAMICS = 'ConditionalFlowMatching',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 PREPROCESS = ['standardize'],
                 BATCH_SIZE = 1024,
                 EPOCHS = 1000,
                 PRINT_EPOCHS = 10,
                 LR = 1e-4,
                 GRADIENT_CLIP = 1.0,
                 DIM_HIDDEN = 300,
                 TIME_EMBEDDING = 'gaussian',
                 DIM_TIME_EMB = 12,
                 DIM_GLOBAL = 16,
                 NUM_EPIC_LAYERS = 20,
                 SIGMA = 1e-5,
                 SOLVER ='midpoint',
                 NUM_SAMPLING_STEPS = 200,
                 DEVICE = 'cuda:3')

cfm.train()

# cfm = Experiment(Config_JetClass_EPiC_CondFlowMatch,
#                  NAME = 'qcd_to_top_gauss',
#                  DATA_SOURCE = 'qcd',
#                  DATA_TARGET = 'top',
#                  DYNAMICS = 'ConditionalFlowMatching',
#                  DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
#                  PREPROCESS = ['standardize'],
#                  BATCH_SIZE = 1024,
#                  WEIGHT_DECAY = 0.00005,   
#                  EPOCHS = 1000,
#                  PRINT_EPOCHS = 10,
#                  LR = 1e-4,
#                  GRADIENT_CLIP = 1.0,
#                  DIM_HIDDEN = 300,
#                  TIME_EMBEDDING = 'gaussian',
#                  DIM_TIME_EMB = 300,
#                  DIM_GLOBAL = 16,
#                  NUM_EPIC_LAYERS = 20,
#                  SIGMA = 1e-5,
#                  SOLVER ='midpoint',
#                  NUM_SAMPLING_STEPS = 200,
#                  DEVICE = 'cuda:0')

# cfm.train()

# cfm = Experiment(Config_JetClass_EPiC_CondFlowMatch,
#                  NAME = 'qcd_to_top_sinus',
#                  DATA_SOURCE = 'qcd',
#                  DATA_TARGET = 'top',
#                  DYNAMICS = 'ConditionalFlowMatching',
#                  DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
#                  PREPROCESS = ['standardize'],
#                  BATCH_SIZE = 1024,
#                  EPOCHS = 1000,
#                  PRINT_EPOCHS = 10,
#                  LR = 1e-4,
#                  GRADIENT_CLIP = 1.0,
#                  DIM_HIDDEN = 256,
#                  TIME_EMBEDDING = 'sinusoidal',
#                  DIM_TIME_EMB = 32,
#                  DIM_GLOBAL = 16,
#                  NUM_EPIC_LAYERS = 16,
#                  SIGMA = 1e-5,
#                  SOLVER ='midpoint',
#                  NUM_SAMPLING_STEPS = 200,
#                  DEVICE = 'cuda:1')

# cfm.train()

# cfm = Experiment(Config_JetClass_EPiC_CondFlowMatch,
#                  NAME = 'qcd_to_top_sinus',
#                  DATA_SOURCE = 'qcd',
#                  DATA_TARGET = 'top',
#                  DYNAMICS = 'ConditionalFlowMatching',
#                  DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
#                  PREPROCESS = ['standardize'],
#                  BATCH_SIZE = 1024,
#                  EPOCHS = 1000,
#                  PRINT_EPOCHS = 10,
#                  LR = 1e-4,
#                  GRADIENT_CLIP = 1.0,
#                  DIM_HIDDEN = 256,
#                  TIME_EMBEDDING = 'sinusoidal',
#                  DIM_TIME_EMB = 256,
#                  DIM_GLOBAL = 16,
#                  NUM_EPIC_LAYERS = 16,
#                  SIGMA = 1e-5,
#                  SOLVER ='midpoint',
#                  NUM_SAMPLING_STEPS = 200,
#                  DEVICE = 'cuda:2')

# cfm.train()