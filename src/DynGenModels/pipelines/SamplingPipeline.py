import torch
from torchdyn.core import NeuralODE
from tqdm.auto import tqdm
from dataclasses import dataclass

from DynGenModels.models.experiment import DefineModel
from DynGenModels.pipelines.utils import TorchdynWrapper

class FlowMatchPipeline:
    
    def __init__(self, 
                 trained_model: DefineModel=None, 
                 preprocessor: object=None,
                 postprocessor: object=None,
                 config: dataclass=None,
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
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model = self.trained_model.best_epoch_model if best_epoch_model else self.trained_model.last_epoch_model

        self.t0 = config.T1 if reverse_time_flow else config.T0
        self.t1 = config.T0 if reverse_time_flow else config.T1
        self.solver = config.SOLVER if solver is None else solver
        self.num_sampling_steps = config.NUM_SAMPLING_STEPS if num_sampling_steps is None else num_sampling_steps
        self.sensitivity = config.SENSITIVITY if sensitivity is None else sensitivity
        self.atol = config.ATOL if atol is None else atol
        self.rtol = config.RTOL if rtol is None else rtol
        self.device = config.DEVICE
        self.time_steps = torch.linspace(self.t0, self.t1, self.num_sampling_steps, device=self.device)
        self.batch_size = config.BATCH_SIZE if batch_size is None else batch_size  

    @torch.no_grad()
    def generate_samples(self, input_source):
        self.source = self._preprocess(input_source)
        self.trajectories = self._postprocess(self._ODEsolver())  

    def _preprocess(self, samples):
        if self.preprocessor is not None:
            self.stats = self.trained_model.dataloader.datasets.summary_stats
            samples = self.preprocessor(samples, methods=self.trained_model.dataloader.datasets.preprocess_methods, summary_stats=self.stats)
            samples.preprocess(format=False)
            return samples.features
        else:
            return samples

    def _postprocess(self, samples):
        if self.postprocessor is not None:
            self.stats = self.trained_model.dataloader.datasets.summary_stats
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
        
        num_batches = self.source.shape[0] // self.batch_size
        if self.source.shape[0] % self.batch_size != 0: num_batches += 1
        trajectories = []
        for i in tqdm(range(num_batches)):
            trajectories.append(node.trajectory(x=self.source[i*self.batch_size:(i+1)*self.batch_size].to(self.device), t_span=self.time_steps).detach().cpu())
        return torch.cat(trajectories, dim=1)

class NormFlowPipeline:
    
    def __init__(self,
                 trained_model: DefineModel=None, 
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