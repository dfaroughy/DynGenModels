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
        
        self.trained_model = trained_model
        self.source = source_input
        self.postprocessor = postprocessor
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model

        self.t0 = configs.t1 if reverse_time_flow else configs.t0
        self.t1 = configs.t0 if reverse_time_flow else configs.t1
        self.solver = configs.solver if solver is None else solver
        self.num_sampling_steps = configs.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        self.sensitivity = configs.sensitivity if sensitivity is None else sensitivity
        self.atol = configs.atol if atol is None else atol
        self.rtol = configs.rtol if rtol is None else rtol
        self.device = configs.DEVICE
        self.time_steps = torch.linspace(self.t0, self.t1, self.num_sampling_steps, device=self.device)
        self.trajectories = self._ODEsolver()

        if self.postprocessor is not None:
            self.stats = self.trained_model.dataloader.datasets.summary_stats if postprocessor is not None else None
            self.postprocess_methods = ['inverse_' + method for method in self.trained_model.dataloader.datasets.preprocess_methods[::-1]]
            print("INFO: post-processing sampled data with {}".format(self.postprocess_methods))

        self.target = self._postprocess(self.trajectories[-1]) if postprocessor is not None else self.trajectories[-1]
        self.midway = self._postprocess(self.trajectories[self.num_sampling_steps // 2]) if postprocessor is not None else self.trajectories[self.num_sampling_steps // 2]
        self.quarter = self._postprocess(self.trajectories[self.num_sampling_steps // 4]) if postprocessor is not None else self.trajectories[self.num_sampling_steps // 4]
        self.thirdquarter = self._postprocess(self.trajectories[3 * self.num_sampling_steps // 4]) if postprocessor is not None else self.trajectories[3 * self.num_sampling_steps // 4]
        self.source = self._postprocess(self.trajectories[0]) if postprocessor is not None else self.trajectories[0]
        
    @torch.no_grad()
    def _ODEsolver(self):
        print('INFO: neural ODE solver with {} method and steps={}'.format(self.solver, self.num_sampling_steps))

        node = NeuralODE(vector_field=TorchdynWrapper(self.model), 
                        solver=self.solver, 
                        sensitivity=self.sensitivity, 
                        seminorm=True if self.solver=='dopri5' else False,
                        atol=self.atol if self.solver=='dopri5' else None, 
                        rtol=self.rtol if self.solver=='dopri5' else None)
        
        if self.source is None:
            assert self.trained_model.dataloader.test is not None, "No test dataset available! provide source input!"
            trajectories = []
            source = self.trained_model.dataloader.test.to(self.device)
            for batch in tqdm(source, desc="sampling"):
                trajectories.append(node.trajectory(x=batch['source'], t_span=self.time_steps))
            trajectories = torch.cat(trajectories, dim=1)
        else:
            trajectories = node.trajectory(x=self.source.to(self.device), t_span=self.time_steps)
        return trajectories.detach().cpu() 

    def _postprocess(self, trajectories):
        sample = self.postprocessor(data=trajectories, 
                                    summary_stats=self.stats, 
                                    methods=self.postprocess_methods)
        sample.postprocess()
        return sample.features



class NormFlowPipeline:
    
    def __init__(self,
                 trained_model: DynGenModelTrainer=None, 
                 preprocessor: object=None,
                 postprocessor: object=None,
                 best_epoch_model: bool=False
                 ):
        
        self.trained_model = trained_model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model

    def _preprocess(self, samples):
        samples = self.preprocessor(samples, methods=self.trained_model.dataloader.datasets.preprocess_methods)
        samples.preprocess()
        return samples.features

    def _postprocess(self, samples):
        self.stats = self.trained_model.dataloader.datasets.summary_stats
        self.postprocess_methods = ['inverse_' + method for method in self.trained_model.dataloader.datasets.preprocess_methods[::-1]]
        samples = self.postprocessor(samples, summary_stats=self.stats, methods=self.postprocess_methods)
        samples.postprocess()
        return samples.features

    @torch.no_grad()
    def generate_samples(self, num: int=1):  
        samples = self.model.sample(num).detach().cpu() 
        self.target = self._postprocess(samples) if self.postprocessor is not None else samples

    @torch.no_grad()     
    def log_prob(self, input):
        input = input.to(self.model.device)
        return self.model.log_prob(self._preprocess(input)).detach().cpu()