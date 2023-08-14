import math
import torch
from typing import Union
from torchdyn.models import NeuralODE

class FlowMatchScheduler:

    def __init__(self, num_train_timesteps=2000, beta_min=0.1, beta_max=20, sampling_eps=1e-3):
        self.sigmas = None
        self.timesteps = None
        self.T = 1.0
        self.eps = sampling_eps

    def ODEsolver(timesteps, 
                source, 
                context, 
                net, 
                solver='midpoint', 
                mask=None , 
                sensitivity="adjoint", 
                atol=1e-4, 
                rtol=1e-4):

        net_wrapped = torchdyn_wrapper(vector_field=net, context=context, mask=mask)

        node = NeuralODE(vector_field=net_wrapped, 
                        solver=solver, 
                        sensitivity=sensitivity, 
                        seminorm=True if solver=='dopri5' else False,
                        atol=atol if solver=='dopri5' else None, 
                        rtol=rtol if solver=='dopri5' else None
                        )

        trajectories = node.trajectory(x=source, t_span=timesteps)
        return trajectories[-1] 

    def set_timesteps(self, num_inference_steps, device: Union[str, torch.device] = None):
        self.timesteps = torch.linspace(self.T, self.eps, num_inference_steps, device=device)

    def step(self):
        # TODO
        pass

    def __len__(self):
        return self.config.num_train_timesteps