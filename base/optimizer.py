import torch
import itertools
from utils.util import get_obj


class Optimizer(object):
    def __init__(
            self,
            optimizer, scheduler=None, grad_clip_norm=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

    def step(self):
        if self.grad_clip_norm is not None:
            params = itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups])
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        self.optimizer.step()

    def step_lr(self, loss, epoch):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def __str__(self):
        format_str = "optimizer param_groups: {0}\nscheduler: {1}\ngrad_clip: {2}\n"
        return format_str.format(
            self.optimizer.state_dict()['param_groups'],
            "None" if self.scheduler is None else self.scheduler.state_dict(),
            self.grad_clip_norm)

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "grad_clip_norm": self.grad_clip_norm
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict.get('optimizer'))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict.get('scheduler'))
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = state_dict.get('grad_clip_norm')


def get_optimizer(optimizer_params, args_dict):
    optimizer_dict = args_dict.pop('optimizer')
    assert optimizer_dict is not None, "Cannot have optimizer none"

    optimizer = get_obj(
        torch.optim,
        optimizer_dict.get('type'),
        optimizer_params,
        **optimizer_dict.get('args'))

    lr_scheduler_dict = args_dict.get('lr_scheduler')
    lr_scheduler = None

    if lr_scheduler_dict is not None:
        lr_scheduler = get_obj(
            torch.optim.lr_scheduler,
            lr_scheduler_dict.get('type'),
            optimizer,
            **lr_scheduler_dict.get('args')
        )

    grad_clip_norm = args_dict.get('grad_clip_norm')

    return Optimizer(
        optimizer=optimizer,
        scheduler=lr_scheduler,
        grad_clip_norm=grad_clip_norm
    )
