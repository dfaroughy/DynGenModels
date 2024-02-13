import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Returns:
    - The Frechet distance between two Gaussians d = ||mu_1 - mu_2||^2 + Trace(sig_1 + sig_2 - 2*sqrt(sig_1 * sig_2)).
    '''
    mse = (mu_1 - mu_2).square().sum(dim=-1)
    trace = sigma_1.trace() + sigma_2.trace() - 2 * torch.linalg.eigvals(sigma_1 @ sigma_2).sqrt().real.sum(dim=-1)
    return mse + trace

@torch.no_grad()
def compute_activation_statistics(model, dataset, batch_size=64, activation_layer='fc1', device='cpu'):
    model.to(device)
    model.eval()
    features = []

    for batch, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        activations = model(batch, activation_layer=activation_layer)

        #...apply global average pooling if features are not 2D

        if len(activations.shape) > 2:
            activations = F.adaptive_avg_pool2d(activations, (1, 1)).view(activations.size(0), -1)

        features.append(activations)

    features = torch.cat(features, dim=0)
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.t())
    return mu, sigma

def compute_fid(model, dataset, dataset_ref=None, mu_ref=None, sigma_ref=None, batch_size=64, activation_layer='fc1', device='cpu'):
    
    assert dataset_ref is not None or (mu_ref is not None and sigma_ref is not None), 'Either dataset_ref or (mu_ref, sigma_ref) must be provided.'
 
    if dataset_ref is None:
        mu, sigma = compute_activation_statistics(model, dataset, batch_size, activation_layer, device)
    else:
        mu_ref, sigma_ref = compute_activation_statistics(model, dataset_ref, batch_size, activation_layer, device)
        mu, sigma = compute_activation_statistics(model, dataset, batch_size, activation_layer, device)

    return frechet_distance(mu, sigma, mu_ref, sigma_ref)