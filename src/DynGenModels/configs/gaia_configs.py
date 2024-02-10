from DynGenModels.configs.utils import Register_Experiment
from DynGenModels.datamodules.gaia.configs import Gaia_Configs
from DynGenModels.models.configs import MLP_Configs, MAF_RQS_Configs, Couplings_RQS_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs
from DynGenModels.dynamics.nf.configs import NormFlow_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs, NormFlows_Sampler_Configs


Gaia_MLP_FlowMatch = Register_Experiment(data = Gaia_Configs,
                                        model = MLP_Configs, 
                                        dynamics = FlowMatch_Configs, 
                                        pipeline = NeuralODE_Sampler_Configs)

Gaia_MAF_RQS_NormFlow = Register_Experiment(data = Gaia_Configs,
                                            model = MAF_RQS_Configs, 
                                            dynamics = NormFlow_Configs, 
                                            pipeline = NormFlows_Sampler_Configs)

Gaia_Couplings_RQS_NormFlow = Register_Experiment(data = Gaia_Configs,
                                                    model = Couplings_RQS_Configs, 
                                                    dynamics = NormFlow_Configs, 
                                                    pipeline = NormFlows_Sampler_Configs)