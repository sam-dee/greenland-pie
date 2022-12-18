import numpy as np
from numpy import ndarray, array

from layers.base_layer import BaseLayer, PHASES


class FCLayer(BaseLayer):
    def __init__(self, inputs, outputs):

        super().__init__(inputs, outputs)
        self.grad_w = None
        self.grad_b = None

    @property
    def trainable(self):
        return True

    def __call__(self, x: ndarray, phase):
        if phase not in PHASES:
            raise ValueError(f'Unknown phase {phase}!')

        y = x.dot(self.w) + self.b

        if phase == 'train':
            self.inputs = x

        return y

    def backward(self, accum_grad):
        # Save weights used during forwards pass
        w = self.w

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            self.grad_w = self.inputs.T.dot(accum_grad)
            self.grad_b = np.sum(accum_grad, axis=0, keepdims=True)

        return accum_grad.dot(w.T)

    def update_weights(self, update_func):
        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        # Update the layer weights
        self.w = update_func(self.w, self.grad_w)
        self.b = update_func(self.b, self.grad_b)

    def get_nrof_trainable_params(self):
        return 0
