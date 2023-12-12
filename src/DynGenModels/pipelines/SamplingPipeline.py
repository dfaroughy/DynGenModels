import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm
from dataclasses import dataclass

from DynGenModels.trainer.trainer import DynGenModelTrainer
from DynGenModels.pipelines.utils import TorchdynWrapper

class FlowMatchPipeline:
    
    def __init__(self, 
                 trained_model: DynGenModelTrainer=None, 
                 preprocessor: object=None,
                 postprocessor: object=None,
                 configs: dataclass=None,
                 solver: str=None,
                 num_sampling_steps: int=None,
                 sensitivity: str=None,
                 atol: float=None,
                 rtol: float=None,
                 reverse_time_flow: bool=False,
                 best_epoch_model: bool=False,
                 batch_size: int=None
                 ):
        
        self.trained_model = trained_model
        self.stats = self.trained_model.dataloader.datasets.summary_stats
        self.preprocessor = preprocessor
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
        self.augmented = configs.augmented
        self.time_steps = torch.linspace(self.t0, self.t1, self.num_sampling_steps, device=self.device)
        self.batch_size = configs.batch_size if batch_size is None else batch_size  

    @torch.no_grad()
    def generate_samples(self, input_source):
        self.source = self._preprocess(input_source)
        self.trajectories = self._postprocess(self._ODEsolver())  
        # self.target = self.trajectories[-1]
        # self.midway = self.trajectories[self.num_sampling_steps // 2]
        # self.quarter = self.trajectories[self.num_sampling_steps // 4]
        # self.thirdquarter = self.trajectories[3 * self.num_sampling_steps // 4]

    def _preprocess(self, samples):
        if self.preprocessor is not None:
            samples = self.preprocessor(samples, methods=self.trained_model.dataloader.datasets.preprocess_methods, summary_stats=self.stats)
            samples.preprocess(format=False)
            return samples.features
        else:
            return samples

    def _postprocess(self, samples):
        if self.postprocessor is not None:
            self.postprocess_methods = ['inverse_' + method for method in self.trained_model.dataloader.datasets.preprocess_methods[::-1]]
            samples = self.postprocessor(samples, methods=self.postprocess_methods, summary_stats=self.stats)
            samples.postprocess()
            return samples.features
        else:
            return samples

    @torch.no_grad()
    def _ODEsolver(self):
        print('INFO: neural ODE solver with {} method and steps={}'.format(self.solver, self.num_sampling_steps))

        node = NeuralODE(vector_field=TorchdynWrapper(self.model), 
                        solver=self.solver, 
                        sensitivity=self.sensitivity, 
                        seminorm=True if self.solver=='dopri5' else False,
                        atol=self.atol if self.solver=='dopri5' else None, 
                        rtol=self.rtol if self.solver=='dopri5' else None)
        
        if self.augmented: 
            self.source = torch.cat([self.source, self.source], dim=-1)
        num_batches = self.source.shape[0] // self.batch_size
        if self.source.shape[0] % self.batch_size != 0: num_batches += 1
        trajectories = []
        for i in tqdm(range(num_batches)):
            trajectories.append(node.trajectory(x=self.source[i*self.batch_size:(i+1)*self.batch_size].to(self.device), t_span=self.time_steps).detach().cpu())
        # trajectories = node.trajectory(x=self.source.to(self.device), t_span=self.time_steps)
        # return trajectories.detach().cpu() 
        return torch.cat(trajectories, dim=1)

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
        if self.preprocessor is not None:
            samples = self.preprocessor(samples, methods=self.trained_model.dataloader.datasets.preprocess_methods)
            samples.preprocess()
            return samples.features
        else:
            return samples

    def _postprocess(self, samples):
        if self.postprocessor is not None:
            self.stats = self.trained_model.dataloader.datasets.summary_stats
            self.postprocess_methods = ['inverse_' + method for method in self.trained_model.dataloader.datasets.preprocess_methods[::-1]]
            samples = self.postprocessor(samples, summary_stats=self.stats, methods=self.postprocess_methods)
            samples.postprocess()
            return samples.features
        else:
            return samples

    @torch.no_grad()
    def generate_samples(self, num: int=1):  
        samples = self.model.sample(num).detach().cpu() 
        self.target = self._postprocess(samples)

    @torch.no_grad()     
    def log_prob(self, input):
        input = input.to(self.model.device)
        return self.model.log_prob(self._preprocess(input)).detach().cpu()