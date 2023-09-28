from typing import Any, List

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import math
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["Schedulers"]


def constant_schedule_with_warmup(optimizer, num_warmup_steps, **kwargs):
    return get_constant_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps
    )


class CosineWithWarmupAndMinLR(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            warmup_fraction = self.last_epoch / self.num_warmup_steps
            return [base_lr * warmup_fraction for base_lr in self.base_lrs]

        progress = 0
        if self.num_training_steps - self.num_warmup_steps > 0:
            progress = (self.last_epoch - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return [max(self.min_lr, base_lr * cosine_decay) for base_lr in self.base_lrs]

class Schedulers:
    """Schedulers factory."""

    _schedulers = {
        "Cosine": CosineWithWarmupAndMinLR,
        #"CosineWithMinLR": get_cosine_schedule_with_warmup_and_min_lr,
        "Linear": get_linear_schedule_with_warmup,
        "Constant": constant_schedule_with_warmup,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._schedulers.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Schedulers.

        Args:
            name: scheduler name
        Returns:
            A class to build the Schedulers
        """
        return cls._schedulers.get(name)
