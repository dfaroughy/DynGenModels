import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm
from dataclasses import dataclass

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.pipelines.utils import TorchdynWrapper

class FlowMatchPipeline:
    
    def __init__(self, 
                 trained_model: DynGenModelTrainer=None, 
                 source_input: torch.Tensor=None,
                 postprocessor: object=None,
                 configs: dataclass=None,
                 solver: str=None,
                 num_sampling_steps: int=None,
                 sensitivity: str=None,
                 atol: float=None,
                 rtol: float=None,
                 reverse_time_flow: bool=False,
                 best_epoch_model: bool=False
                 ):
        
        self.model = trained_model
        self.source = source_input
        self.postprocessor = postprocessor
        self.net = self.model.best_epoch_model if best_epoch_model else self.model.last_epoch_model

        self.t0 = configs.t1 if reverse_time_flow else configs.t0
        self.t1 = configs.t0 if reverse_time_flow else configs.t1
        self.solver = configs.solver if solver is None else solver
        self.num_sampling_steps = configs.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        self.sensitivity = configs.sensitivity if sensitivity is None else sensitivity
        self.atol = configs.atol if atol is None else atol
        self.rtol = configs.rtol if rtol is None else rtol

        self.time_steps = torch.linspace(self.t0, self.t1, self.num_sampling_steps)
        self.trajectories = self.ODEsolver()

        if self.postprocessor is not None:
            self.stats = self.model.dataloader.datasets.summary_stats if postprocessor is not None else None
            self.postprocess_methods = ['inverse_' + method for method in self.model.dataloader.datasets.preprocess_methods[::-1]]
            print("INFO: post-processing sampled data with {}".format(self.postprocess_methods))

        self.target = self.postprocess(self.trajectories[-1]) if postprocessor is not None else self.trajectories[-1]
        self.source = self.postprocess(self.trajectories[0]) if postprocessor is not None else self.trajectories[0]

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

    def postprocess(self, trajectories):
        sample = self.postprocessor(data=trajectories, summary_stats=self.stats, methods=self.postprocess_methods)
        sample.postprocess()
        return sample.features


class NormFlowPipeline:
    
    def __init__(self,
                 trained_model: DynGenModelTrainer=None, 
                 postprocessor: object=None,
                 configs: dataclass=None,
                 num_gen_samples: int=None,
                 best_epoch_model: bool=False
                 ):
        
        self.model = trained_model
        self.num_gen_samples = configs.num_gen_samples if num_gen_samples is None else num_gen_samples
        self.postprocessor = postprocessor
        self.net = self.model.best_epoch_model if best_epoch_model else self.model.last_epoch_model
        self.samples = self.sampler()
        
        if self.postprocessor is not None:
            self.stats = self.model.dataloader.datasets.summary_stats if postprocessor is not None else None
            self.postprocess_methods = ['inverse_' + method for method in self.model.dataloader.datasets.preprocess_methods[::-1]]
            print("INFO: post-processing sampled data with {}".format(self.postprocess_methods))

        self.target = self.postprocess(self.samples) if postprocessor is not None else self.samples

    @torch.no_grad()
    def sampler(self):
        return self.model.dynamics.flow.sample(self.num_gen_samples).detach()

    def postprocess(self, samples):
        sample = self.postprocessor(data=samples, summary_stats=self.stats, methods=self.postprocess_methods)
        sample.postprocess()
        return sample.features