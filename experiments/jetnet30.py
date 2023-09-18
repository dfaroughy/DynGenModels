from DynGenModels.datamodules.jetnet.datasets import JetNetDataset
from DynGenModels.datamodules.jetnet.dataloader import JetNetDataLoader
from DynGenModels.models.deepsets import DeepSets
from DynGenModels.configs.deepsets_config import DeepSetsConfig as Config
from DynGenModels.trainer.trainer import FlowMatchTrainer
from DynGenModels.dynamics.flowmatch import SimpleCFM

config = Config(features    = ['eta_rel', 'phi_rel', 'pt_rel', 'e_rel',  'R'],
                preprocess  = ['standardize'],
                datasets    = {'jetnet30' : ['t30.hdf5', 'particle_features']},
                labels      = {'jetnet30' : 0},
                data_split_fracs = [0.6, 0.1, 0.3],
                max_num_jets=None,
                max_num_constituents=30,
                epochs = 1000,
                batch_size = 1024,
                warmup_epochs= 50,
                dim_hidden = 256, 
                num_layers_1 = 3,
                num_layers_2 = 3,
                device = 'cpu'
                )

root_dir =  '/home/df630/' if 'cuda' in config.device else '/Users/dario/Dropbox/PROJECTS/ML/'
root_dir += 'DynGenModels'

if __name__ == "__main__":

    deepsets = DeepSets(model_config=config)
    config.set_workdir(root_dir + '/results', save_config=True)
    datasets = JetNetDataset(dir_path = root_dir + '/data/jetnet', 
                            datasets = config.datasets,
                            class_labels = config.labels,
                            max_num_jets = config.max_num_jets,
                            max_num_constituents = config.max_num_constituents,
                            preprocess = config.preprocess,
                            particle_features = config.features,
                            remove_negative_pt = True
                            ) 
    dataloader = JetNetDataLoader(datasets=datasets, data_split_fracs=config.data_split_fracs, batch_size=config.batch_size)

    for b in dataloader.train:
        print(b)
        break

    dynamics = SimpleCFM(model=deepsets)
    model = FlowMatchTrainer(dynamics=dynamics, dataloader=dataloader)

    model.train()