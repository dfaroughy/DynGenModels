from DynGenModels.configs.registered_experiments import Config_MNIST_Unet28x28_CondFlowMatch 
from DynGenModels.models.experiment import Experiment

cfm = Experiment(Config_MNIST_Unet28x28_CondFlowMatch,
                 NAME = 'emnist_to_mnist',
                 DATA_SOURCE = 'emnist',
                 DATA_TARGET = 'mnist',
                 DYNAMICS = 'OptimalTransportCFM',
                 MODEL = 'Unet28x28',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 BATCH_SIZE = 256,
                 EPOCHS = 20,
                 LR = 5e-3,
                 DIM_HIDDEN = 32, 
                 DIM_TIME_EMB = 32, 
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 ATOL = 1e-4,
                 RTOL = 1e-4,
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')

cfm.train()

cfm = Experiment(Config_MNIST_Unet28x28_CondFlowMatch,
                 NAME = 'emnist_to_mnist',
                 DATA_SOURCE = 'emnist',
                 DATA_TARGET = 'mnist',
                 DYNAMICS = 'OptimalTransportCFM',
                 MODEL = 'Unet28x28',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 BATCH_SIZE = 256,
                 EPOCHS = 20,
                 LR = 1e-3,
                 DIM_HIDDEN = 32, 
                 DIM_TIME_EMB = 32, 
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 ATOL = 1e-4,
                 RTOL = 1e-4,
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')

cfm.train()

cfm = Experiment(Config_MNIST_Unet28x28_CondFlowMatch,
                 NAME = 'emnist_to_mnist',
                 DATA_SOURCE = 'emnist',
                 DATA_TARGET = 'mnist',
                 DYNAMICS = 'OptimalTransportCFM',
                 MODEL = 'Unet28x28',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 BATCH_SIZE = 256,
                 EPOCHS = 20,
                 LR = 5e-4,
                 DIM_HIDDEN = 32, 
                 DIM_TIME_EMB = 32, 
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 ATOL = 1e-4,
                 RTOL = 1e-4,
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')

cfm.train()

cfm = Experiment(Config_MNIST_Unet28x28_CondFlowMatch,
                 NAME = 'emnist_to_mnist',
                 DATA_SOURCE = 'emnist',
                 DATA_TARGET = 'mnist',
                 DYNAMICS = 'OptimalTransportCFM',
                 MODEL = 'Unet28x28',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 BATCH_SIZE = 128,
                 EPOCHS = 20,
                 LR = 1e-4,
                 DIM_HIDDEN = 32, 
                 DIM_TIME_EMB = 32, 
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 ATOL = 1e-4,
                 RTOL = 1e-4,
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')

cfm.train()

cfm = Experiment(Config_MNIST_Unet28x28_CondFlowMatch,
                 NAME = 'emnist_to_mnist',
                 DATA_SOURCE = 'emnist',
                 DATA_TARGET = 'mnist',
                 DYNAMICS = 'OptimalTransportCFM',
                 MODEL = 'Unet28x28',
                 DATA_SPLIT_FRACS = [0.8, 0.2, 0.0],
                 BATCH_SIZE = 128,
                 EPOCHS = 20,
                 LR = 5e-5,
                 DIM_HIDDEN = 32, 
                 DIM_TIME_EMB = 32, 
                 SIGMA = 0.0,
                 SOLVER ='midpoint',
                 ATOL = 1e-4,
                 RTOL = 1e-4,
                 NUM_SAMPLING_STEPS = 1000,
                 DEVICE = 'cuda:2')

cfm.train()