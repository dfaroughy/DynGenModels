
from DynGenModels.configs.utils import Configs
from DynGenModels.datamodules.deconvolution.configs import Deconvolution_Gauss_Configs
from DynGenModels.models.configs import MLP_Configs, MAF_RQS_Configs, MAF_Affine_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs
from DynGenModels.dynamics.nf.configs import Deconvolution_NormFlow_Configs
from DynGenModels.pipelines.configs import NormFlows_Sampler_Configs, NeuralODE_Sampler_Configs


Deconvolution_Gauss_MLP_FlowMatch = Configs(data = Deconvolution_Gauss_Configs,
                                     model = MLP_Configs, 
                                     dynamics = FlowMatch_Configs, 
                                     pipeline = NeuralODE_Sampler_Configs)


Deconvolution_Gauss_MAF_Affine_NormFlow = Configs(data = Deconvolution_Gauss_Configs,
                                           model = MAF_Affine_Configs, 
                                           dynamics = Deconvolution_NormFlow_Configs, 
                                           pipeline = NormFlows_Sampler_Configs)


Deconvolution_Gauss_MAF_RQS_NormFlow = Configs(data = Deconvolution_Gauss_Configs,
                                        model = MAF_RQS_Configs, 
                                        dynamics = Deconvolution_NormFlow_Configs, 
                                        pipeline = NormFlows_Sampler_Configs)

