
import json
from datetime import datetime
from dataclasses import dataclass, asdict, make_dataclass, field, fields, MISSING

from DynGenModels.utils.utils import make_dir, print_table

def DynGenModelConfigs(*dataclasses):
    combined_fields = []

    for dc in dataclasses:
        for f in fields(dc):
            if f.default is MISSING and f.default_factory is MISSING:
                combined_fields.append((f.name, f.type))
            elif f.default_factory is not MISSING:
                # We define a new field with the default factory instead of using the function directly
                combined_fields.append((f.name, f.type, field(default_factory=f.default_factory)))
            else:
                combined_fields.append((f.name, f.type, f.default))

    Combined = make_dataclass("Configs", combined_fields, bases=(object,))

    def set_workdir(self, path: str = '.', dir_name: str = None, save_config: bool = True):
        time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
        dir_name = '{}.{}_{}'.format(self.model_name, self.data_name, time) if dir_name is None else dir_name
        self.workdir = make_dir(path + '/' + dir_name, overwrite=False)
        if save_config: self.save()

    def save(self, path: str = None):
        config = asdict(self)
        print_table(config)
        path = self.workdir + '/config.json' if path is None else path
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: config = json.load(json_file)
        print_table(config)
        return cls(**config)

    # Attach methods to class
    setattr(Combined, "set_workdir", set_workdir)
    setattr(Combined, "save", save)
    setattr(Combined, "load", load)

    return Combined



# @dataclass
# class Toys_FlowMatch_MLP_Configs(_FlowMatchingConfigs, 
#                                 _SamplingConfigs,
#                                 _TrainingConfigs, 
#                                 _DataConfigs):

#     model_name : str = 'Toys_FlowMatch_MLP'
#     dim_input  : int = 3 
#     dim_hidden : int = 128   

#     def __post_init__(self):
#         self.dim_input = len(self.features)

#     def set_workdir(self, path: str='.', dir_name: str=None, save_config: bool=True):
#         time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
#         dir_name = '{}.{}_{}'.format(self.model_name, self.data_name, time) if dir_name is None else dir_name
#         self.workdir = make_dir(path + '/' + dir_name, overwrite=False)
#         if save_config: self.save()

#     def save(self, path: str=None):
#         config = asdict(self)
#         print_table(config)
#         path = self.workdir + '/config.json' if path is None else path
#         with open(path, 'w') as f:
#             json.dump(config, f, indent=4)

#     @classmethod
#     def load(cls, path: str):
#         with open(path, 'r') as json_file: config = json.load(json_file)
#         print_table(config)
#         return cls(**config)


