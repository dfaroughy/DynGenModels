from DynGenModels.configs.registered_experiments import Config_JetClass_EPiC_CondFlowMatch, Config_JetClass_DeepSets_CondFlowMatch
from DynGenModels.models.experiment import Experiment

cfm = Experiment(Config_JetClass_DeepSets_CondFlowMatch,
                 NAME = 'qcd_to_top',
                 DATA_SOURCE = 'qcd',
                 DATA_TARGET = 'top',
                 DYNAMICS = 'ConditionalFlowMatching',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 PREPROCESS = ['standardize'],
                 BATCH_SIZE = 256,
                 EPOCHS = 5000,
                 EARLY_STOPPING = 20,
                 MIN_EPOCHS = 300,
                 PRINT_EPOCHS = 10,
                 LR = 1e-4,
                 DIM_HIDDEN = 128,
                 DIM_TIME_EMB = 16,
                 ACTIVATION = 'ReLU',
                 NUM_LAYERS_PHI = 3,
                 NUM_LAYERS_RHO = 3,
                 DROPOUT = 0.1,
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')


cfm.train()

cfm = Experiment(Config_JetClass_DeepSets_CondFlowMatch,
                 NAME = 'qcd_to_top',
                 DATA_SOURCE = 'qcd',
                 DATA_TARGET = 'top',
                 DYNAMICS = 'OptimalTransportFlowMatching',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 PREPROCESS = ['standardize'],
                 BATCH_SIZE = 256,
                 EPOCHS = 5000,
                 EARLY_STOPPING = 20,
                 MIN_EPOCHS = 300,
                 PRINT_EPOCHS = 10,
                 LR = 1e-4,
                 DIM_HIDDEN = 128,
                 DIM_TIME_EMB = 16,
                 ACTIVATION = 'ReLU',
                 NUM_LAYERS_PHI = 3,
                 NUM_LAYERS_RHO = 3,
                 DROPOUT = 0.1,
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')

cfm.train()
