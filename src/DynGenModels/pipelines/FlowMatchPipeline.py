import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm

from DynGenModels.trainer.trainer import FlowMatchTrainer
from DynGenModels.pipelines.utils import TorchdynWrapper

class FlowMatchPipeline:
    
    def __init__(self, 
                 trained_model: FlowMatchTrainer=None, 
                 source_input: torch.Tensor=None,
                 postprocessor: object=None,
                 solver: str='euler',
                 num_sampling_steps: int=100,
                 sensitivity: str='adjoint',
                 atol: float=1e-4,
                 rtol: float=1e-4):
        
        self.model = trained_model
        self.source = source_input
        self.postprocessor = postprocessor
        self.net = self.model.dynamics.net
        self.stats = self.model.dataloader.datasets.summary_stats
        self.postprocess_methods = ['inverse_' + method for method in self.model.dataloader.datasets.preprocess_methods[::-1]]
        self.time = torch.linspace(self.model.dynamics.T, 0, num_sampling_steps)
        self.solver = solver
        self.sensitivity = sensitivity
        self.atol = atol
        self.rtol = rtol
        self.trajectories = self.ODEsolver()

        print("INFO: post-processing sampled data with {}".format(self.postprocess_methods))
        self.target = self.postprocess(self.trajectories[-1]) if postprocessor is not None else self.trajectories[-1]

    @torch.no_grad()
    def ODEsolver(self):
        node = NeuralODE(vector_field=TorchdynWrapper(self.net), 
                        solver=self.solver, 
                        sensitivity=self.sensitivity, 
                        seminorm=True if self.solver=='dopri5' else False,
                        atol=self.atol if self.solver=='dopri5' else None, 
                        rtol=self.rtol if self.solver=='dopri5' else None
                        )
        if self.source is None:
            trajectories = []
            for batch in tqdm(self.model.dataloader.test, desc="sampling"):
                trajectories.append(node.trajectory(x=batch['source'], t_span=self.time))
            trajectories = torch.cat(trajectories, dim=1)
        else:
            trajectories = node.trajectory(x=self.source, t_span=self.time)
        return trajectories

    def postprocess(self, trajectories):
        sample = self.postprocessor(data=trajectories, summary_stats=self.stats, methods=self.postprocess_methods)
        sample.postprocess()
        return sample.galactic_features
