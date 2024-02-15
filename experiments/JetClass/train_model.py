from DynGenModels.configs.registered_experiments import Config_JetClass_EPiC_CondFlowMatch, Config_JetClass_DeepSets_CondFlowMatch
from DynGenModels.models.experiment import Experiment

cfm = Experiment(Config_JetClass_DeepSets_CondFlowMatch,
                 NAME = 'qcd_to_top',
                 DATA_SOURCE = 'qcd',
                 DATA_TARGET = 'top',
                 DYNAMICS = 'ConditionalFlowMatching',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 PREPROCESS = ['normalize', 'logit_transform', 'standardize'],
                 BATCH_SIZE = 128,
                 EPOCHS = 100,
                 LR = 1e-4,
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 NUM_SAMPLING_STEPS = 100,
                 DEVICE = 'cuda:1')

cfm.train()