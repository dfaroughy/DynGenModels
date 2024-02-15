from DynGenModels.configs.registered_experiments import Config_MNIST_UNet_CondFlowMatch 
from DynGenModels.models.experiment import Experiment

cfm = Experiment(Config_MNIST_UNet_CondFlowMatch,
                 NAME = 'emnist_to_mnist',
                 DATA_SOURCE = 'emnist',
                 DATA_TARGET = 'mnist',
                 DYNAMICS = 'OptimalTransportFlowMatching',
                 MODEL = 'Unet',
                 DATA_SPLIT_FRACS = [1.0, 0.0, 0.0],
                 BATCH_SIZE = 128,
                 EPOCHS = 100,
                 LR = 1e-4,
                 DIM_HIDDEN = 32, 
                 SIGMA = 0.0,
                 SOLVER ='dopri5',
                 ATOL = 1e-4,
                 RTOL = 1e-4,
                 NUM_SAMPLING_STEPS = 100,
                 DEVICE = 'cuda:3')

cfm.train()