
from DynGenModels.configs.utils import Configs

from DynGenModels.datamodules.fermi.configs import FermiGCE_Configs
from DynGenModels.models.configs import MLP_Configs, ResNet_Configs, MAF_RQS_Configs, MAF_Affine_Configs, Couplings_RQS_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs, CondFlowMatch_Configs
from DynGenModels.dynamics.nf.configs import NormFlow_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs, NormFlows_Sampler_Configs


FermiGCE_MLP_FlowMatch = Configs(data = FermiGCE_Configs,
                                 model = MLP_Configs, 
                                 dynamics = FlowMatch_Configs, 
                                 pipeline = NeuralODE_Sampler_Configs)


FermiGCE_MLP_CondFlowMatch = Configs(data = FermiGCE_Configs,
                                     model = MLP_Configs, 
                                     dynamics = CondFlowMatch_Configs, 
                                     pipeline = NeuralODE_Sampler_Configs)


FermiGCE_ResNet_FlowMatch = Configs(data = FermiGCE_Configs,
                                   model = ResNet_Configs, 
                                   dynamics = FlowMatch_Configs, 
                                   pipeline = NeuralODE_Sampler_Configs)


FermiGCE_ResNet_CondFlowMatch = Configs(data = FermiGCE_Configs,
                                       model = ResNet_Configs, 
                                       dynamics = CondFlowMatch_Configs, 
                                       pipeline = NeuralODE_Sampler_Configs)


FermiGCE_MAF_Affine_NormFlow = Configs(data = FermiGCE_Configs,
                                      model = MAF_Affine_Configs, 
                                      dynamics = NormFlow_Configs, 
                                      pipeline = NormFlows_Sampler_Configs)


FermiGCE_MAF_RQS_NormFlow = Configs(data = FermiGCE_Configs,
                                    model = MAF_RQS_Configs, 
                                    dynamics = NormFlow_Configs, 
                                    pipeline = NormFlows_Sampler_Configs)


FermiGCE_Couplings_RQS_NormFlow = Configs(data = FermiGCE_Configs,
                                          model = Couplings_RQS_Configs, 
                                          dynamics = NormFlow_Configs, 
                                          pipeline = NormFlows_Sampler_Configs)