import json
from datetime import datetime
from typing import Type
from dataclasses import dataclass, asdict, make_dataclass, field, fields, MISSING
from DynGenModels.utils.utils import make_dir, print_table

def Load_Experiment_Config(data: Type[dataclass], model: Type[dataclass], dynamics: Type[dataclass], pipeline: Type[dataclass]=None):  # type: ignore
    
    combined_fields = []
    all_dataclasses = [data, model, dynamics] 
    all_dataclasses += [pipeline] if pipeline is not None else []

    for dc in all_dataclasses:
        for f in fields(dc):
            if f.default is MISSING and f.default_factory is MISSING: combined_fields.append((f.name, f.type))
            elif f.default_factory is not MISSING: combined_fields.append((f.name, f.type, field(default_factory=f.default_factory)))
            else: combined_fields.append((f.name, f.type, f.default))

    Combined = make_dataclass("Configs", combined_fields, bases=(object,))

    def set_workdir(self, path: str, dir_name: str=None, save_config: bool=True):  # type: ignore
        time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
        dir_name = '{}.{}.{}.{}'.format(self.NAME, self.DYNAMICS, self.MODEL, time) if dir_name is None else dir_name
        self.WORKDIR = make_dir(path + '/' + dir_name, overwrite=True)
        if save_config: self.save()

    def save(self, path: str=None, print: bool=True): # type: ignore
        config = asdict(self)
        if print: print_table(config)
        path = self.WORKDIR + '/config.json' if path is None else path
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