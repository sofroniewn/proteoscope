import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_value: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    if current_step > num_training_steps:
        return min_value

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

    # Cosine decay from 1.0 to min_value
    cosine_decay = (1.0 + math.cos(math.pi * progress)) / 2.0 * (1.0 - min_value) + min_value

    return cosine_decay


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, 
    last_epoch: int = -1, min_value: float = 0.0
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_value, after a warmup period during which it increases linearly between 
    0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_value (`float`, *optional*, defaults to 0.0):
            The minimal value the scheduler will decay towards.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_value=min_value,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
