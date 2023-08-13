
import pytest
import torch

#...architecture

@pytest.mark.parametrize("B", [1, 10]) # batch size
@pytest.mark.parametrize("P", [1, 10]) # num points

def test_jetnet_dataloader():
    pass
 
