import numpy as np
import torch
import math
# Copied from https://github.com/asteroid-team/asteroid/blob/master/asteroid/engine/schedulers.py
# Copied since it is the last function we still use from asteroid (and avoid other dependencies)
class BaseScheduler(object):
    '''Base class for the step-wise scheduler logic.
    Args:
        optimizer (Optimize): Optimizer instance to apply lr schedule on.
    Subclass this and overwrite ``_get_lr`` to write your own step-wise scheduler.
    '''
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0
    def zero_grad(self):
        self.optimizer.zero_grad()
    def _get_lr(self):
        raise NotImplementedError
    def _set_lr(self, lr):
        for l, param_group in zip(lr, self.optimizer.param_groups):
            param_group["lr"] = l
    def step(self, metrics=None, epoch=None):
        '''Update step-wise learning rate before optimizer.step.'''
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}
    def as_tensor(self, start=0, stop=100_000):
        '''Returns the scheduler values from start to stop.'''
        lr_list = []
        for _ in range(start, stop):
            self.step_num += 1
            lr_list.append(self._get_lr())
        self.step_num = 0
        return torch.tensor(lr_list)
    def plot(self, start=0, stop=100_000):  # noqa
        '''Plot the scheduler values from start to stop.'''
        import matplotlib.pyplot as plt
        all_lr = self.as_tensor(start=start, stop=stop)
        plt.plot(all_lr.numpy())
        plt.show()

class ExponentialWarmup(BaseScheduler):
    """ Scheduler to apply ramp-up during training to the learning rate.
    Args:
        optimizer: torch.optimizer.Optimizer, the optimizer from which to rampup the value from
        max_lr: float, the maximum learning to use at the end of ramp-up.
        rampup_length: int, the length of the rampup (number of steps).
        exponent: float, the exponent to be used.
    """

    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0, tot_steps=None, min_lr_scaler=0.1, cos_down=True):
        super().__init__(optimizer)
        self.rampup_len = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent
        self.tot_steps = tot_steps
        self.min_lr_scaler = min_lr_scaler
        self.ema_steps = self.rampup_len * 5
        self.cos_down = cos_down

    def _get_scaling_factor(self):

        if self.rampup_len == 0:
            return 1.0
        else:
            current = np.clip(self.step_num, 0.0, self.rampup_len)
            phase = 1.0 - current / self.rampup_len
            return float(np.exp(self.exponent * phase * phase))

    def _get_exp_up_cos_down(self, exponent=None):
        if exponent is None:
            exponent = self.exponent
        if self.step_num < self.rampup_len:
            current = np.clip(self.step_num, 0.0, self.rampup_len)
            phase = 1.0 - current / self.rampup_len
            return float(np.exp(exponent * phase * phase))
        else:
            current = np.clip(self.step_num, self.rampup_len, self.step_num)
            phase = (current - self.rampup_len) / (self.tot_steps - self.rampup_len) * math.pi
            return (math.cos(phase) + 1) / 2   

    def _get_lr(self):
        if self.cos_down:
            if self.step_num < self.rampup_len:
                lr = [lr * self._get_scaling_factor() for lr in self.max_lr]
            else:
                lr = [(x - x * self.min_lr_scaler) * self._get_exp_up_cos_down() + x * self.min_lr_scaler for x in self.max_lr]
            return lr
        else:
            lr = [lr * self._get_scaling_factor() for lr in self.max_lr] # + [lr * self._get_exp_up_cos_down() for lr in self.max_lr[2:]]
            return lr
