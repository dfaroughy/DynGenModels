
import pytest
import torch

#...architecture

@pytest.mark.parametrize("B", [1, 10]) # batch size
@pytest.mark.parametrize("P", [1, 10]) # num points
@pytest.mark.parametrize("D", [1, 3]) # feature dimension
@pytest.mark.parametrize("C", [1, 2]) # context dimension
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
@pytest.mark.parametrize("mask_type", ["ones", "random"])

def test_deepset_architecture(B, P, D, C, mask_type, device):

    from DynGenModels.models.deepsets import DeepSetNet

    B = 5 # batch size
    P = 10 # number of particles
    
    time = torch.rand((B, P, 1))
    features = torch.rand((B, P, D))
    context = torch.rand((B, P, C))

    if mask_type == "ones": mask = torch.ones((B, P))
    elif mask_type == "random": 
        mask = torch.rand((B, P))
        mask = (mask > 0.5).float()  # Convert to a binary mask

    model = DeepSetNet(dim=D, dim_context=C, device=device)
    output = model(t=time, x=features, context=context, mask=mask)
    assert True

#...wrapper and config

@pytest.mark.parametrize("dim_context", [0, 1, 2]) 
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
@pytest.mark.parametrize("mask_type", ["ones", "random"])

def test_deepset_wrapper_and_config(dim_context, mask_type, device):

    from DynGenModels.models.deepsets import DeepSets
    from DynGenModels.trainer.configs.deepsets_config import DeepSetsConfig as config

    config.dim_context = dim_context
    config.device = device

    B = config.batch_size 
    P = config.max_num_constituents
    D = config.dim_input
    C = config.dim_context

    time = torch.rand((B, P, 1))
    features = torch.rand((B, P, D))
    context = torch.rand((B, P, C))

    if mask_type == "ones": mask = torch.ones((B, P))
    elif mask_type == "random": 
        mask = torch.rand((B, P))
        mask = (mask > 0.5).float()  # Convert to a binary mask

    model = DeepSets(config)
    print(model)
    output = model(time, features, context, mask)
    assert True
