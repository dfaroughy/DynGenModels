import pytest

from DynGenModels.datamodules.jetnet.datasets import JetNetDataset

SAMPLE_DIR_PATH = '../data/jetnet30' 
SAMPLE_DATASETS = {'jetnet' : ['t.h5', 'particle features']}  
SAMPLE_CLASS_LABELS = {'jetnet' : -1}  
PARTICLE_FEATURES = ['eta_rel', 'phi_rel', 'pt_rel', 'R']
PREPROCESS = ['standardize'] 
MAX_NUM_JETS = 10
MAX_NUM_PARTICLES = 30
REMOVE_NEGATIVE_PT = True
COMPUTE_JET_FEATURES =  False


# 1. Instantiation Test
def test_instantiation():
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, datasets=SAMPLE_DATASETS, class_labels=SAMPLE_CLASS_LABELS, particle_features=PARTICLE_FEATURES, preprocess=PREPROCESS, max_num_jets=MAX_NUM_JETS, max_num_particles=MAX_NUM_PARTICLES, remove_negative_pt=REMOVE_NEGATIVE_PT, compute_jet_features=COMPUTE_JET_FEATURES)
    assert isinstance(dataset, JetNetDataset)

# 2. Length Test
def test_len():
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, datasets=SAMPLE_DATASETS, class_labels=SAMPLE_CLASS_LABELS, particle_features=PARTICLE_FEATURES, preprocess=PREPROCESS, max_num_jets=MAX_NUM_JETS, max_num_particles=MAX_NUM_PARTICLES, remove_negative_pt=REMOVE_NEGATIVE_PT, compute_jet_features=COMPUTE_JET_FEATURES)
    assert len(dataset) == MAX_NUM_JETS  

# 3. Data Retrieval Test
def test_data_retrieval():
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, datasets=SAMPLE_DATASETS, class_labels=SAMPLE_CLASS_LABELS, particle_features=PARTICLE_FEATURES, preprocess=PREPROCESS, max_num_jets=MAX_NUM_JETS, max_num_particles=MAX_NUM_PARTICLES, remove_negative_pt=REMOVE_NEGATIVE_PT, compute_jet_features=COMPUTE_JET_FEATURES)
    data = dataset[0]
    assert 'label' in data
    assert 'mask' in data
    assert 'particle_features' in data

# 4. Iterability Test
def test_iterability():
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, datasets=SAMPLE_DATASETS, class_labels=SAMPLE_CLASS_LABELS, particle_features=PARTICLE_FEATURES, preprocess=PREPROCESS, max_num_jets=MAX_NUM_JETS, max_num_particles=MAX_NUM_PARTICLES, remove_negative_pt=REMOVE_NEGATIVE_PT, compute_jet_features=COMPUTE_JET_FEATURES)
    for data in dataset:
        assert 'label' in data

# 5. Saving and Loading Test
def test_save_and_load(tmpdir):
    dataset = JetNetDataset(dir_path=SAMPLE_DIR_PATH, datasets=SAMPLE_DATASETS, class_labels=SAMPLE_CLASS_LABELS, particle_features=PARTICLE_FEATURES, preprocess=PREPROCESS, max_num_jets=MAX_NUM_JETS, max_num_particles=MAX_NUM_PARTICLES, remove_negative_pt=REMOVE_NEGATIVE_PT, compute_jet_features=COMPUTE_JET_FEATURES)
    save_path = tmpdir.mkdir("data").join("sample_save_path")
    dataset.save(str(save_path))
    loaded_dataset = JetNetDataset.load(str(save_path))
    assert isinstance(loaded_dataset, JetNetDataset)
    assert len(loaded_dataset) == len(dataset)
