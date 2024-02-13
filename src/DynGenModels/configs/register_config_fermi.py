
from DynGenModels.configs.utils import Load_Experiment_Config

from DynGenModels.datamodules.fermi.configs import FermiGCE_Config
from DynGenModels.models.architectures.configs import MLP_Config, ResNet_Config, MAF_RQS_Config, MAF_Affine_Config, Couplings_RQS_Config
from DynGenModels.dynamics.cnf.configs import FlowMatch_Config, CondFlowMatch_Config
from DynGenModels.dynamics.nf.configs import NormFlow_Config
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Config, NormFlows_Sampler_Config


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