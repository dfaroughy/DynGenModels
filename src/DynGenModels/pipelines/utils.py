import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from DynGenModels.trainer.trainer import FlowMatchTrainer

class RunFlowPipeline:
    def __init__(self,
                 workdir: str,
                 dataset: torch.utils.data,
                 dataloader: torch.utils.data,  
                 net: torch.nn.Module,
                 configs: dataclass,
                 dynamics: object,
                 pipeline: object,
                 postprocessor: object=None,
                 source_input: torch.Tensor=None):

        configs.set_workdir(workdir, save_config=True)
        self.configs=configs
        self.postprocessor=postprocessor
        self.source_input=source_input

        self.net = net(configs)
        self.dataset = dataset(configs) 
        self.dataloader = dataloader(self.dataset, configs)

        self.model = FlowMatchTrainer(dynamics=dynamics(self.net, configs), 
                                      dataloader=self.dataloader, 
                                      config=configs)
        self.model.train()
        self.pipeline = pipeline(trained_model=self.model, 
                                postprocessor=self.postprocessor, 
                                config=self.configs,
                                source_input=self.source_input)
                                
        self.target = self.pipeline.target
        self.source = self.pipeline.source
        self.trajectories = self.pipeline.trajectories

class TorchdynWrapper(torch.nn.Module):
    """ Wraps model to torchdyn compatible format.
    """
    def __init__(self, net):
        super().__init__()
        self.nn = net
    def forward(self, t, x):
        t = t.repeat(x.shape[:-1]+(1,), 1)
        return self.nn(t=t, x=x)
