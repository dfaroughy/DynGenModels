import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class BinaryClassifierTest:

    def __init__(self, configs: dataclass):
        self.device = configs.DEVICE
        self.criterion = nn.BCELoss()


    def loss(self, model, batch):
        features = batch['data'].to(self.device)
        labels = batch['labels'].to(self.device)
        logits = model(features)
        loss = self.criterion(logits, labels)
        return loss
    
    def predict(self, model, batch):
        features = batch['data'].to(self.device)
        probs = model(features)
        # probs = F.softmax(logits, dim=1)
        return probs.detach().cpu() 