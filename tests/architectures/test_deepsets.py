
import pytest
import torch


@pytest.mark.parametrize("D", [1, 3]) # feature dimension
@pytest.mark.parametrize("C", [1, 2]) # context dimension
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
@pytest.mark.parametrize("mask_type", ["ones", "random"])

def test_deepset_architecture(D, C, mask_type, device):
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

    #...Eval model 
    model = DeepSetNet(dim=D, dim_context=C, device=device)

    #...Forward pass
    output = model(t=time, x=features, context=context, mask=mask)

    #...Loss
    target = torch.rand((B, P, D))
    criterion = torch.nn.MSELoss()
    loss = criterion(output, target)

    #...Backprop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('loss={}, output.shape={}'.format(loss.item(), output.shape))
    assert True

def test_deepset_wrapper_and_config():
    from DynGenModels.models.deepsets import DeepSets
    from DynGenModels.trainer.configs.deepsets_config import DeepSetsConfig
    config = DeepSetsConfig()
    model = DeepSets(config)
    assert True
