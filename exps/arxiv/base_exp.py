import torch
from abc import ABCMeta, abstractmethod
from typing import Dict
from torch.nn import Module


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment.
    """

    def __init__(self, args):
        self.seed = None
        self.output_dir = args.log_path
        self.print_interval = 1000
        self.eval_interval = 10

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(self, batch_size: int, is_distributed: bool) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass
