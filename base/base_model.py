from abc import abstractmethod
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
        self._trainable_params = None

    @property
    def trainable_params(self):
        if self._trainable_params is None:
            self._trainable_params = filter(
                lambda p: p.requires_grad, self.parameters())
        return self._trainable_params

    def __str__(self):
        model_parameters = self.trainable_params
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
