from DynGenModels.configs.utils import Load_Experiment_Config
from DynGenModels.dynamics.cnf.configs import FlowMatch_Config, CondFlowMatch_Config
from DynGenModels.dynamics.nf.configs import NormFlow_Config
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Config, NormFlows_Sampler_Config
from DynGenModels.models.architectures.configs import (MLP_Config, ResNet_Config,
                                                       DeepSets_Config, EPiC_Config,
                                                       UNet_Config, UNetLight_Config,
                                                       MAF_RQS_Config, MAF_Affine_Config, 
                                                       Couplings_RQS_Config)

###########################################
### Registered Toy experiments configs ####
###########################################

from DynGenModels.datamodules.toys.configs import Gauss_2_Moons_Config, Gauss_2_Gauss_Config

Config_Gauss_2_Moons_MLP_FlowMatch = Load_Experiment_Config(data = Gauss_2_Moons_Config,
                                                            model = MLP_Config, 
                                                            dynamics = FlowMatch_Config, 
                                                            pipeline = NeuralODE_Sampler_Config)
Config_Gauss_2_Gauss_MLP_FlowMatch = Load_Experiment_Config(data = Gauss_2_Gauss_Config,
                                                            model = MLP_Config, 
                                                            dynamics = FlowMatch_Config, 
                                                            pipeline = NeuralODE_Sampler_Config)

#############################################
### Registered MNIST experiments configs ####
#############################################

from DynGenModels.datamodules.mnist.configs import MNIST_Config

Config_MNIST_UNet_FlowMatch = Load_Experiment_Config(data = MNIST_Config,
                                                     model = UNet_Config, 
                                                     dynamics = FlowMatch_Config, 
                                                     pipeline = NeuralODE_Sampler_Config)
Config_MNIST_UNet_CondFlowMatch = Load_Experiment_Config(data = MNIST_Config,
                                                         model = UNet_Config, 
                                                         dynamics = CondFlowMatch_Config, 
                                                         pipeline = NeuralODE_Sampler_Config)
Config_MNIST_UNetLight_CondFlowMatch = Load_Experiment_Config(data = MNIST_Config,
                                                              model = UNetLight_Config, 
                                                              dynamics = CondFlowMatch_Config, 
                                                              pipeline = NeuralODE_Sampler_Config)

#################################################
### Registered Fermi GEC experiments configs ####
#################################################

from DynGenModels.datamodules.fermi.configs import FermiGCE_Config

Config_FermiGCE_MLP_FlowMatch = Load_Experiment_Config(data = FermiGCE_Config,
                                                       model = MLP_Config, 
                                                       dynamics = FlowMatch_Config, 
                                                       pipeline = NeuralODE_Sampler_Config)
Config_FermiGCE_MLP_CondFlowMatch = Load_Experiment_Config(data = FermiGCE_Config,
                                                           model = MLP_Config, 
                                                           dynamics = CondFlowMatch_Config, 
                                                           pipeline = NeuralODE_Sampler_Config)
Config_FermiGCE_ResNet_FlowMatch = Load_Experiment_Config(data = FermiGCE_Config,
                                                          model = ResNet_Config, 
                                                          dynamics = FlowMatch_Config, 
                                                          pipeline = NeuralODE_Sampler_Config)
Config_FermiGCE_ResNet_CondFlowMatch = Load_Experiment_Config(data = FermiGCE_Config,
                                                              model = ResNet_Config, 
                                                              dynamics = CondFlowMatch_Config, 
                                                              pipeline = NeuralODE_Sampler_Config)
Config_FermiGCE_MAF_Affine_NormFlow = Load_Experiment_Config(data = FermiGCE_Config,
                                                             model = MAF_Affine_Config, 
                                                             dynamics = NormFlow_Config, 
                                                             pipeline = NormFlows_Sampler_Config)
Config_FermiGCE_MAF_RQS_NormFlow = Load_Experiment_Config(data = FermiGCE_Config,
                                                          model = MAF_RQS_Config, 
                                                          dynamics = NormFlow_Config, 
                                                          pipeline = NormFlows_Sampler_Config)
Config_FermiGCE_Couplings_RQS_NormFlow = Load_Experiment_Config(data = FermiGCE_Config,
                                                                model = Couplings_RQS_Config, 
                                                                dynamics = NormFlow_Config, 
                                                                pipeline = NormFlows_Sampler_Config)

############################################
### Registered Gaia experiments configs ####
############################################

from DynGenModels.datamodules.gaia.configs import Gaia_Config

Config_Gaia_MLP_FlowMatch = Load_Experiment_Config(data = Gaia_Config,
                                                   model = MLP_Config, 
                                                   dynamics = FlowMatch_Config, 
                                                   pipeline = NeuralODE_Sampler_Config)
Config_Gaia_MAF_RQS_NormFlow = Load_Experiment_Config(data = Gaia_Config,
                                                      model = MAF_RQS_Config, 
                                                      dynamics = NormFlow_Config, 
                                                      pipeline = NormFlows_Sampler_Config)
Config_Gaia_Couplings_RQS_NormFlow = Load_Experiment_Config(data = Gaia_Config,
                                                            model = Couplings_RQS_Config, 
                                                            dynamics = NormFlow_Config, 
                                                            pipeline = NormFlows_Sampler_Config)

############################################
### Registered JetNet experiments configs ##
############################################

from DynGenModels.datamodules.jetnet.configs import JetNet_Config

Config_JetNet_DeepSets_FlowMatch = Load_Experiment_Config(data = JetNet_Config,
                                                          model = DeepSets_Config, 
                                                          dynamics = FlowMatch_Config, 
                                                          pipeline = NeuralODE_Sampler_Config)
Config_JetNet_DeepSets_CondFlowMatch = Load_Experiment_Config(data = JetNet_Config,
                                                              model = DeepSets_Config, 
                                                              dynamics = CondFlowMatch_Config, 
                                                              pipeline = NeuralODE_Sampler_Config)
Config_JetNet_EPiC_CondFlowMatch = Load_Experiment_Config(data = JetNet_Config,
                                                          model = EPiC_Config, 
                                                          dynamics = CondFlowMatch_Config, 
                                                          pipeline = NeuralODE_Sampler_Config)

############################################
### Registered JetClass experiments configs ##
############################################

from DynGenModels.datamodules.jetclass.configs import JetClass_Config

Config_JetClass_MLP_CondFlowMatch = Load_Experiment_Config(data = JetClass_Config,
                                                            model = MLP_Config, 
                                                            dynamics = CondFlowMatch_Config, 
                                                            pipeline = NeuralODE_Sampler_Config)
Config_JetClass_DeepSets_CondFlowMatch = Load_Experiment_Config(data = JetClass_Config,
                                                                model = DeepSets_Config, 
                                                                dynamics = CondFlowMatch_Config, 
                                                                pipeline = NeuralODE_Sampler_Config)
Config_JetClass_EPiC_CondFlowMatch = Load_Experiment_Config(data = JetClass_Config,
                                                            model = EPiC_Config, 
                                                            dynamics = CondFlowMatch_Config, 
                                                            pipeline = NeuralODE_Sampler_Config)






