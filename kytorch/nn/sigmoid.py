from kytorch import Tensor, CallableGradFn, TrainableWeight, Gradient, GradFn
from .module import Module

import numpy as np


class SigmoidGradFn(CallableGradFn):

    def compute_gradient(
        self, loss_wrt_child: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        x = self.input_tensor.data
        child_wrt_input = np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        loss_wrt_input = Gradient(loss_wrt_child.data * child_wrt_input)
        return loss_wrt_input, []


class Sigmoid(Module):

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-x.data)), SigmoidGradFn(x))
