from DynGenModels.configs.utils import Load_Experiment_Config

# Register Gaia experiments configs:

from DynGenModels.datamodules.gaia.configs import Gaia_Config
from DynGenModels.models.architectures.configs import MLP_Config, MAF_RQS_Config, Couplings_RQS_Config
from DynGenModels.dynamics.cnf.configs import FlowMatch_Config
from DynGenModels.dynamics.nf.configs import NormFlow_Config
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Config, NormFlows_Sampler_Config


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