import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from DynGenModels.trainer.trainer import FlowMatchTrainer

# Create a mock dataset and mock model for testing

data = torch.randn(100, 2)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class MockDynamics(nn.Module):
    def __init__(self):
        super(MockDynamics, self).__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)

    def loss(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        return torch.nn.functional.cross_entropy(outputs, targets)

@pytest.fixture
def mock_dataloader():
    # Split the mock data into train and valid datasets
    class DataLoaderWrapper:
        def __init__(self):
            self.train = dataloader
            self.valid = dataloader
    return DataLoaderWrapper()

def test_trainer_instantiation(mock_dataloader):
    trainer = FlowMatchTrainer(MockDynamics(), mock_dataloader)
    assert isinstance(trainer.model, MockDynamics)
    assert trainer.epochs == 100

def test_trainer_run(mock_dataloader):
    trainer = FlowMatchTrainer(MockDynamics(), mock_dataloader, epochs=2, early_stopping=None)
    trainer.train()
    assert len(trainer.model.fc.weight) == 2

# More tests can be added as needed.
