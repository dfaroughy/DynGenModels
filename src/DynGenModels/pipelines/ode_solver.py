import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm
from DynGenModels.pipelines.utils import TorchdynWrapper

@torch.no_grad()
def ODEsolver(self):
    print('INFO: neural ODE solver with {} method and steps={}'.format(self.solver, self.num_sampling_steps))

    node = NeuralODE(vector_field=TorchdynWrapper(self.net), 
                    solver=self.solver, 
                    sensitivity=self.sensitivity, 
                    seminorm=True if self.solver=='dopri5' else False,
                    atol=self.atol if self.solver=='dopri5' else None, 
                    rtol=self.rtol if self.solver=='dopri5' else None
                    )
    if self.source is None:
        assert self.model.dataloader.test is not None, "No test dataset available! provide source input!"
        trajectories = []
        for batch in tqdm(self.model.dataloader.test, desc="sampling"):
            trajectories.append(node.trajectory(x=batch['source'], t_span=self.time_steps))
        trajectories = torch.cat(trajectories, dim=1)
    else:
        trajectories = node.trajectory(x=self.source, t_span=self.time_steps)
    return trajectories