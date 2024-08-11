from kytorch import Tensor, CallableGradFn, TrainableWeight, Gradient, GradFn
from dataclasses import dataclass
from .module import Module

import numpy as np


@dataclass
class LinearGradFn(CallableGradFn):
    weight: TrainableWeight
    bias: TrainableWeight

    def compute_gradient(
        self, loss_wrt_child: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        child_wrt_weight = Gradient(self.input_tensor.data.T)
        child_wrt_bias = Gradient(np.ones_like(self.bias.data))

        loss_wrt_input = Gradient(loss_wrt_child.data @ self.weight.data)
        loss_wrt_weight = Gradient(np.outer(loss_wrt_child.data, child_wrt_weight.data))
        loss_wrt_bias = Gradient(loss_wrt_child.data * child_wrt_bias.data)

        return loss_wrt_input, [
            (loss_wrt_weight, self.weight),
            (loss_wrt_bias, self.bias),
        ]


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = TrainableWeight(
            np.random.randn(out_features, in_features).astype(np.float32)
        )
        self.bias = TrainableWeight(np.random.randn(out_features).astype(np.float32))

    def forward(self, x: Tensor) -> Tensor:
        assert x.is_finite(), f"Non-finite values in input: {x.data}"

        forward_result = self.weight.data @ x.data + self.bias.data
        result_tensor = Tensor(forward_result, LinearGradFn(x, self.weight, self.bias))
        return result_tensor

    # def backward(self, x: np.ndarray) -> LinearPartials:
    #     output_wrt_weight = x
    #     output_wrt_bias = np.ones_like(self.bias)
    #     return LinearPartials(output_wrt_weight, output_wrt_bias)
