import torch
from torchdyn.core import NeuralODE

from DynGenModels.datamodules.fermi.process import PostprocessData
from DynGenModels.trainer.trainer import FlowMatchTrainer
from DynGenModels.pipelines.utils import TorchdynWrapper

class FlowMatchPipeline:
    
    def __init__(self, 
                 source_data: torch.Tensor=None,
                 pretrained_model: FlowMatchTrainer=None, 
                 solver: str='euler',
                 sampling_steps: int=100):
        
        super().__init__()

        self.source = source_data
        self.net = pretrained_model.dynamics.model
        self.stats = pretrained_model.dataloader.datasets.summary_statistics['dataset']
        self.postprocess = ['inverse_' + method for method in pretrained_model.dataloader.datasets.preprocess_methods[::-1]]
        self.time = torch.linspace(0, 1, sampling_steps)
        self.solver = solver
        self.trajectories = self.ODEsolver()
        self.target = self.trajectories[-1]

    @torch.no_grad()
    def ODEsolver(self, sensitivity="adjoint", atol=1e-4, rtol=1e-4):
        node = NeuralODE(vector_field=TorchdynWrapper(self.net), 
                        solver=self.solver, 
                        sensitivity=sensitivity, 
                        seminorm=True if self.solver=='dopri5' else False,
                        atol=atol if self.solver=='dopri5' else None, 
                        rtol=rtol if self.solver=='dopri5' else None
                        )
        trajectories = node.trajectory(x=self.source, t_span=self.time)
        return self.post_process(trajectories)

    def post_process(self, trajectories):
        sample = PostprocessData(data=trajectories, stats=self.stats, methods=self.postprocess)
        sample.postprocess()
        return sample.galactic_features
