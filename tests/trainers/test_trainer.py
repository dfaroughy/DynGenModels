
import pytest
import torch

#...architecture

@pytest.mark.parametrize("B", [1, 10]) # batch size
@pytest.mark.parametrize("P", [1, 10]) # num points
@pytest.mark.parametrize("D", [1, 3]) # feature dimension
@pytest.mark.parametrize("C", [1, 2]) # context dimension
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
@pytest.mark.parametrize("mask_type", ["ones", "random"])

def test_trainer( device):

    from DynGenModels.models.deepsets import DeepSet
