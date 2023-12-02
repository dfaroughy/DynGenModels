
from DynGenModels.dynamics.nf.normflows import NormalizingFlow
from DynGenModels.models.nflow_nets import MAFPiecewiseRQS 
from DynGenModels.configs.fermi_configs import FermiGCE_MAF_RQS_NormFlow as Configs

configs = Configs(# data params:
                 DATA = 'FermiGCE',
                 dataset = '../../data/fermi/fermi_test2_PS.npy',
                 features = ['theta', 'phi', 'energy'],
                 preprocess = ['normalize', 'logit_transform', 'standardize'],
                 cuts = {'theta': [5., 25.], 'phi': [5., 25.], 'energy': [1000, 10000]},
                 data_split_fracs = [0.8, 0.2, 0.0],
                 # training params:
                 DEVICE = 'cuda:1',
                 EPOCHS = 10000,
                 batch_size = 15000,
                 print_epochs = 20,
                 early_stopping = 100,
                 min_epochs = 500,
                 lr = 1e-4,
                 optimizer = 'Adam',
                fix_seed = None,
                 # dynamics params:
                 DYNAMICS = 'NormFlow',
                 permutation = 'reverse',
                 num_transforms = 3,
                 # model params:
                 num_blocks = 3,
                 dim_hidden = 256, 
                 dropout = 0.1,
                 num_bins = 20,
                 tail_bound = 20, 
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

maf = DynGenModelTrainer(dynamics = NormalizingFlow(configs), 
                         model = MAFPiecewiseRQS(configs), 
                         dataloader = FermiDataLoader(fermi, configs), 
                         configs = configs)

#...train model:

maf.train()

#...sample from model:

from DynGenModels.pipelines.SamplingPipeline import NormFlowPipeline
from DynGenModels.datamodules.fermi.dataprocess import PreProcessFermiData, PostProcessFermiData 

pipeline = NormFlowPipeline(trained_model=maf, 
                            preprocessor=PreProcessFermiData,
                            postprocessor=PostProcessFermiData,
                            best_epoch_model=True)

#...plot results:


pipeline.generate_samples(num=fermi.target.shape[0])


from utils import results_plots, results_2D_plots

results_plots(data=fermi.target, 
              generated=pipeline.target, 
              comparator='pull',
              model = configs.MODEL, 
              save_path=configs.workdir + '/fermi_features.pdf', 
              bins=150, 
              features=[r'$\theta$', r'$\phi$', r'$E$ [GeV]'])

results_2D_plots(pipeline.target,
                 save_path=configs.workdir + '/fermi_features_2D.pdf',  
                 gridsize=200)
