
from DynGenModels.dynamics.nf.normflows import NormalizingFlow
from DynGenModels.models.nflow_nets import CouplingsPiecewiseRQS 
from DynGenModels.configs.fermi_configs import FermiGCE_Couplings_RQS_NormFlow as Configs

configs = Configs(# data params:
                 DATA = 'FermiGCE',
                 dataset = '../../data/fermi/fermi_data_galactic_coord.npy',
                 features = ['theta', 'phi', 'energy'],
                 preprocess = ['normalize', 'logit_transform', 'standardize'],
                 cuts = {'theta': [-10., 10.], 'phi': [-5., 10.], 'energy': [1000, 2000]},
                 data_split_fracs = [0.8, 0.2, 0.0],
                 # training params:
                 DEVICE = 'cuda:2',
                 EPOCHS = 10000,
                 batch_size = 15000,
                 print_epochs = 20,
                 early_stopping = 100,
                 min_epochs = 1000,
                 lr = 1e-4,
                 optimizer = 'Adam',
                 fix_seed = None,
                 # dynamics params:
                 DYNAMICS = 'NormFlow',
                 permutation = '1-cycle',
                 mask = 'checkerboard',
                 num_transforms = 10,
                 # model params:
                 num_blocks = 3,
                 dim_hidden = 256, 
                 dropout = 0.1,
                 num_bins = 30,
                 tail_bound = 10, 
                 use_residual_blocks = False,
                 use_batch_norm = False
                 )

#...set working directory for results:

configs.set_workdir(path='../../results', save_config=True)

#...define dataset:

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.datamodules.fermi.datasets import FermiDataset 
from DynGenModels.datamodules.fermi.dataloader import FermiDataLoader 

fermi = FermiDataset(configs)
nf = DynGenModelTrainer(dynamics = NormalizingFlow(configs), 
                        model = CouplingsPiecewiseRQS(configs), 
                        dataloader = FermiDataLoader(fermi, configs), 
                        configs = configs)

#...train model:

nf.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import NormFlowPipeline
from DynGenModels.datamodules.fermi.dataprocess import PostProcessFermiData 

pipeline = NormFlowPipeline(trained_model=nf, 
                            configs=configs, 
                            postprocessor=PostProcessFermiData,
                            num_gen_samples=fermi.target.shape[0],
                            best_epoch_model=False)

pipeline_best = NormFlowPipeline(trained_model=nf, 
                                 configs=configs, 
                                 postprocessor=PostProcessFermiData,
                                 num_gen_samples=fermi.target.shape[0],
                                 best_epoch_model=True)


#...plot results:

from utils import results_plots, results_2D_plots

results_plots(data=fermi.target, 
              generated=pipeline.target, 
              comparator='pull',
              model = configs.MODEL, 
              save_path=configs.workdir + '/fermi_features.pdf', 
              bins=300, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])

results_plots(data=fermi.target, 
              generated=pipeline_best.target, 
              comparator='pull',
              model = configs.MODEL, 
              save_path=configs.workdir + '/fermi_features_best.pdf', 
              bins=300, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])

results_2D_plots(pipeline.target,
                 save_path=configs.workdir + '/fermi_features_2D.pdf',  
                 gridsize=200)

results_2D_plots(pipeline_best.target,
                 save_path=configs.workdir + '/fermi_features_2D_best.pdf',  
                 gridsize=200)