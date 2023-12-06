
from DynGenModels.configs.utils import Configs

from DynGenModels.datamodules.cathode.configs import Cathode_Configs
from DynGenModels.models.configs import MLP_Classifier_Configs
from DynGenModels.metrics.configs import Classifier_Configs


Cathode_MLP_Classifier = Configs(data = Cathode_Configs,
                                model = MLP_Classifier_Configs, 
                                dynamics = Classifier_Configs, 
                                pipeline = None)
