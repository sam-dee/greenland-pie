import numpy as np
from numpy import ndarray, array

from layers.activation_functions import ReLU, Sigmoid
from layers.base_layer import BaseLayer, PHASES

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid
}


class ActivationLayer(BaseLayer):
    def __init__(self, inputs, outputs, name='relu'):
        super().__init__(inputs, outputs)

        if name not in activation_functions:
            raise ValueError(f'Unknown activation function {name}!')
        self.activation = activation_functions[name]()

    @property
    def trainable(self):
        return True

    def __call__(self, x: ndarray, phase):
        if phase not in PHASES:
            raise ValueError(f'Unknown phase {phase}!')

        if phase == 'train':
            self.inputs = x

        return self.activation(x)

    def backward(self, dy):
        return dy * self.activation.gradient(self.inputs)

    def get_nrof_trainable_params(self):
        return np.prod(self.w.shape) + np.prod(self.b.shape)
