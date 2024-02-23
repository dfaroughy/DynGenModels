from DynGenModels.configs.registered_experiments import Config_JetClass_EPiC_CondFlowMatch
from DynGenModels.models.experiment import Experiment

cfm = Experiment(Config_JetClass_EPiC_CondFlowMatch,
                 NAME = 'noise_to_top',
                 DATA_SOURCE = 'noise',
                 DATA_TARGET = 'top',
                 MAX_NUM_CONSTITUENTS = 30,
                 DYNAMICS = 'ContextSchrodingerBridgeCFM',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 PREPROCESS = ['standardize'],
                 BATCH_SIZE = 1024,
                 EPOCHS = 1000,
                 PRINT_EPOCHS = 10,
                 LR = 1e-4,
                 GRADIENT_CLIP = 1.0,
                 DIM_HIDDEN = 300,
                 TIME_EMBEDDING = 'sinusoidal',
                 USE_SKIP_CONNECTIONS = True,
                 DIM_TIME_EMB = 12,
                 DIM_GLOBAL = 16,
                 NUM_EPIC_LAYERS = 20,
                 SIGMA = 1e-5,
                 SOLVER ='midpoint',
                 NUM_SAMPLING_STEPS = 200,
                 DEVICE = 'cuda:2')

cfm.train()

from DynGenModels.datamodules.jetclass.dataprocess import PostProcessJetClassData as Postprocessor
from utils import plot_consitutents, plot_jets 

cfm.generate_samples(cfm.dataset.source_preprocess, Postprocessor=Postprocessor)
plot_consitutents(cfm, save_dir=cfm.config.WORKDIR, features=[r'$p^{\rm rel}_T$', r'$\Delta\eta$', r'$\Delta\phi$'], figsize=(12,3.5))
plot_jets(cfm, save_dir=cfm.config.WORKDIR, features=[r'$p_t$', r'$\eta$', r'$\phi$', r'$m$'], figsize=(12,3))
