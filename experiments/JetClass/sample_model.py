import sys

from DynGenModels.configs.registered_experiments import Config_JetClass_EPiC_CondFlowMatch
from DynGenModels.models.experiment import Experiment
from DynGenModels.datamodules.jetclass.dataprocess import PostProcessJetClassData as Postprocessor

from utils import plot_consitutents, plot_jets 

path = '../../results/' + sys.argv[2]
cfm = Experiment(Config_JetClass_EPiC_CondFlowMatch, path=path, DEVICE='cuda:'+str(sys.argv[1]))
cfm.load(model='best')
cfm.generate_samples(cfm.dataset.source_preprocess[:10000], Postprocessor=Postprocessor)
plot_consitutents(cfm, save_dir=path, features=[r'$p^{\rm rel}_T$', r'$\Delta\eta$', r'$\Delta\phi$'], figsize=(12,3.5))
plot_jets(cfm, save_dir=path, plot_source=False, features=[r'$p_t$', r'$\eta$', r'$\phi$', r'$m$'], figsize=(12,3))