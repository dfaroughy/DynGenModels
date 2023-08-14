
import pytest
from DynGenModels.datamodules.jetnet.dataloader import JetNetDataLoader
from DynGenModels.datamodules.jetnet.datasets import JetNetDataset

SAMPLE_DIR_PATH = '/Users/dario/Dropbox/PROJECTS/ML/DynGenModels/data/jetnet' 
SAMPLE_DATASETS = {'jetnet30': ['t30.hdf5', 'particle_features'], 'jetnet150': ['t150.hdf5', 'particle_features']}  
SAMPLE_CLASS_LABELS = {'jetnet30': 0, 'jetnet150': 1}  
DATASET = JetNetDataset(dir_path=SAMPLE_DIR_PATH, 
                        datasets=SAMPLE_DATASETS, 
                        class_labels=SAMPLE_CLASS_LABELS, 
                        max_num_jets=1000, 
                        preprocess=['compute_jet_features', 'standardize'])
FRACS = [0.5, 0.2, 0.3]
BATCH_SIZE = 8

#...TEST 1: Instantiation 

def test_datadataloader_initialization():
    dataloader = JetNetDataLoader(datasets=DATASET, data_split_fracs=FRACS, batch_size=BATCH_SIZE)
    assert dataloader.train is not None
    assert dataloader.valid is not None
    assert dataloader.test is not None

#...Test 2: Splitting Test

@pytest.mark.parametrize("FRACS", [[0.4, 0.3, 0.3], [0.6, 0.3, 0.1]])
@pytest.mark.parametrize("SAMPLE_DATASETS, SAMPLE_CLASS_LABELS", [({'jetnet30': ['t30.hdf5', 'particle_features']}, 
                                                                    {'jetnet30': 0}) , 
                                                                   ({'jetnet30': ['t30.hdf5', 'particle_features'], 
                                                                     'jetnet150': ['t150.hdf5', 'particle_features']}, 
                                                                    {'jetnet30': 0, 
                                                                     'jetnet150': 1})])

def test_datadataloader_split(FRACS, SAMPLE_DATASETS, SAMPLE_CLASS_LABELS):
    _DATASET = JetNetDataset(dir_path=SAMPLE_DIR_PATH, 
                            datasets=SAMPLE_DATASETS, 
                            class_labels=SAMPLE_CLASS_LABELS, 
                            max_num_jets=1000, 
                            preprocess=['compute_jet_features', 'standardize'])
    dataloader = JetNetDataLoader(datasets=_DATASET, data_split_fracs=FRACS, batch_size=BATCH_SIZE)
    assert len(dataloader.test.dataset) == int(len(_DATASET) * (1 - FRACS[0] - FRACS[1]))
    assert len(dataloader.train.dataset) == int(len(_DATASET) * FRACS[0])
    assert len(dataloader.valid.dataset) == int(len(_DATASET) * FRACS[1])

#...Test 3: looping over batch

@pytest.mark.parametrize("BATCH_SIZE", [8, 16])

def test_datadataloader_data(BATCH_SIZE):
    dataloader = JetNetDataLoader(datasets=DATASET, data_split_fracs=FRACS, batch_size=BATCH_SIZE)
    for batch in dataloader.train:
        assert 'label' in batch 
        assert 'mask' in batch
        assert 'particle_features' in batch
        assert 'jet_features' in batch
        assert batch['label'].size(0) == BATCH_SIZE
        assert batch['mask'].size(0) == BATCH_SIZE
        assert batch['particle_features'].size(0) == BATCH_SIZE
        assert batch['jet_features'].size(0) == BATCH_SIZE
        break
    for batch in dataloader.test:
        assert 'label' in batch 
        assert 'mask' in batch
        assert 'particle_features' in batch
        assert 'jet_features' in batch
        assert batch['label'].size(0) == BATCH_SIZE
        assert batch['mask'].size(0) == BATCH_SIZE
        assert batch['particle_features'].size(0) == BATCH_SIZE
        assert batch['jet_features'].size(0) == BATCH_SIZE
        break
    