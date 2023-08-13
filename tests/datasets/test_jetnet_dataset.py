import pytest
from DynGenModels.datamodules.jetnet.datasets import JetNetDataset

SAMPLE_DIR_PATH = '/Users/dario/Dropbox/PROJECTS/ML/DynGenModels/data/jetnet' 
SAMPLE_DATASETS = {'jetnet30' : ['t30.hdf5', 'particle_features'], 'jetnet150' : ['t150.hdf5', 'particle_features']}  
SAMPLE_CLASS_LABELS = {'jetnet30' : 0, 'jetnet150' : 1}  
PARTICLE_FEATURES = ['eta_rel', 'phi_rel', 'pt_rel']
PREPROCESS = None
MAX_NUM_JETS = 10
MAX_NUM_CONSTITUENTS = 20
REMOVE_NEGATIVE_PT = True

#...1 Instantiation Test
def test_instantiation():
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, datasets=SAMPLE_DATASETS, class_labels=SAMPLE_CLASS_LABELS, max_num_jets=MAX_NUM_JETS)
    assert isinstance(dataset, JetNetDataset)

#...2 Iterability Test
@pytest.mark.parametrize("PREPROCESS", [None, ['compute_jet_features', 'standardize']]) 
def test_iterability(PREPROCESS):
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, 
                            datasets=SAMPLE_DATASETS, 
                            class_labels=SAMPLE_CLASS_LABELS, 
                            particle_features=PARTICLE_FEATURES,
                            preprocess=PREPROCESS, 
                            max_num_jets=MAX_NUM_JETS,
                            max_num_constituents=MAX_NUM_CONSTITUENTS,
                            remove_negative_pt=REMOVE_NEGATIVE_PT)
    for jet in dataset:
        assert 'label' in jet
        assert 'mask' in jet
        assert 'particle_features' in jet
        if PREPROCESS is not None and 'compute_jet_features' in PREPROCESS: assert 'jet_features' in jet
        else: assert 'jet_features' not in jet

#...3 Length Test
@pytest.mark.parametrize("MAX_NUM_JETS", [10, 100]) 
def test_len(MAX_NUM_JETS):
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, 
                            datasets=SAMPLE_DATASETS, 
                            class_labels=SAMPLE_CLASS_LABELS, 
                            particle_features=PARTICLE_FEATURES,
                            preprocess=PREPROCESS, 
                            max_num_jets=MAX_NUM_JETS,
                            max_num_constituents=MAX_NUM_CONSTITUENTS,
                            remove_negative_pt=REMOVE_NEGATIVE_PT)
    assert len(dataset) == MAX_NUM_JETS * len(SAMPLE_DATASETS)

#...4 Data retrieval and shape Test
@pytest.mark.parametrize("PARTICLE_FEATURES",  [['eta_rel', 'phi_rel', 'pt_rel'], ['eta_rel', 'phi_rel', 'pt_rel', 'e_rel', 'R']])
@pytest.mark.parametrize("MAX_NUM_CONSTITUENTS",  [20, 30])
def test_data_shapesl(PARTICLE_FEATURES, MAX_NUM_CONSTITUENTS, PREPROCESS=['compute_jet_features']):
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, 
                            datasets=SAMPLE_DATASETS, 
                            class_labels=SAMPLE_CLASS_LABELS, 
                            particle_features=PARTICLE_FEATURES,
                            preprocess=PREPROCESS, 
                            max_num_jets=MAX_NUM_JETS,
                            max_num_constituents=MAX_NUM_CONSTITUENTS,
                            remove_negative_pt=REMOVE_NEGATIVE_PT)
    jet = dataset[0]
    assert jet['label'] in SAMPLE_CLASS_LABELS.values()
    assert jet['mask'].shape == (MAX_NUM_CONSTITUENTS,)
    assert jet['particle_features'].shape == (MAX_NUM_CONSTITUENTS, len(PARTICLE_FEATURES)) 
    if PREPROCESS is not None and 'compute_jet_features' in PREPROCESS: assert jet['jet_features'].shape == (5,)
