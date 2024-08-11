from kytorch import Tensor, CallableGradFn, TrainableWeight, Gradient, GradFn
from .module import Module

import numpy as np


class SinGradFn(CallableGradFn):

    def compute_gradient(
        self, loss_wrt_child: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        x = self.input_tensor.data
        child_wrt_input = np.cos(x)
        loss_wrt_input = Gradient(loss_wrt_child.data * child_wrt_input)
        return loss_wrt_input, []


class Sin(Module):

    def forward(self, x: Tensor) -> Tensor:
        # Use the value as the angle in radians
        sin_result = np.sin(x.data)
        return Tensor(sin_result, SinGradFn(x))
