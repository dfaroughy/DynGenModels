import pytest
import torch
from DynGenModels.dynamics.flowmatch import SimpleCFM

# Define a fixture to generate mock tensors for testing.
@pytest.fixture
def mock_tensors():
    B = 10 # Batch size
    P = 5 # Number of particles
    D = 3 # Dimension of the particle features
    C = 2 # Dimension of the context
    return {
        'time': torch.randn((B, P, 1)),
        'target': torch.randn((B, P, D)),
        'source': torch.randn((B, P, D)),
        'context': torch.randn((B, P, C)),
        'mask': torch.ones((B, P, 1))
    }

def mock_model(t, x, context, mask):
    return torch.randn_like(x)

# Test the initialization and basic properties of SimpleCFM
def test_simplecfm_initialization():
    model = SimpleCFM()
    assert model.sigma_min == 1e-6

# Test the z function, probability path function and the conditional vector field u
def test_dynamics_components(mock_tensors):
    model = SimpleCFM()
    model.z(mock_tensors['source'], mock_tensors['target'], mock_tensors['context'])
    assert torch.all(model.x0 == mock_tensors['source'])
    assert torch.all(model.x1 == mock_tensors['target'])
    mean, std = model.probability_path(mock_tensors['time'])
    expected_mean = mock_tensors['time'] * model.x1 + (1 - mock_tensors['time']) * model.x0
    assert torch.all(mean == expected_mean)
    assert std == model.sigma_min
    u = model.cond_vector_field(mock_tensors['source'], mock_tensors['time'])
    assert torch.all(u == mock_tensors['target'] - mock_tensors['source'])
    
# Test the loss function of ConditionalFlowMatching
def test_conditionalflowmatching_loss(mock_tensors):
    model = SimpleCFM()
    loss = model.loss(mock_tensors['time'], mock_tensors['target'], mock_tensors['source'], mock_tensors['context'], mock_tensors['mask'], mock_model)
    assert loss.dim() == 0  # Check that the loss is a scalar
    assert loss >= 0  # Loss should be non-negative
