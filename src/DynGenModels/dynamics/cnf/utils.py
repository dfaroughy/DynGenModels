
import warnings
import numpy as np
import ot as pot
import torch

class OTPlanSampler:

    ''' modified version of torchcfm.utils 
    '''

    def __init__(self, reg: float = 0.05, reg_m: float = 1.0, normalize_cost: bool = False, warn: bool = True):
        self.ot_fn = pot.emd
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=False):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=False):
        pi = self.get_map(x0, x1)
        self.i, self.j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[self.i], x1[self.j]
