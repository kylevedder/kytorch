from kytorch import Tensor, CallableGradFn, TrainableWeight, Gradient, GradFn
from .module import Module

import numpy as np


class ReluGradFn(CallableGradFn):

    def compute_gradient(
        self, loss_wrt_child: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        x = self.input_tensor.data
        child_wrt_input = (x >= 0).astype(x.dtype)
        loss_wrt_input = Gradient(loss_wrt_child.data * child_wrt_input)
        return loss_wrt_input, []


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:
        relu_result = np.maximum(0, x.data)
        return Tensor(relu_result, ReluGradFn(x))
