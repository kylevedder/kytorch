import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TrainableWeight:
    data: np.ndarray

    def __post_init__(self):
        assert self.is_finite(), "Non-finite weight created"

    def update(self, data: np.ndarray):
        self.data = data
        self.__post_init__()

    def is_finite(self) -> bool:
        return np.all(np.isfinite(self.data))


@dataclass
class Gradient:
    data: np.ndarray

    def __post_init__(self):
        assert np.isfinite(self.data).all(), "Non-finite gradient created"


class LeafGradFn:

    def backwards(self, child_grad: Gradient) -> list[tuple[Gradient, TrainableWeight]]:
        return []


@dataclass
class Tensor:
    data: np.ndarray
    grad_fn: "GradFn" = LeafGradFn()

    def __post_init__(self):
        assert self.is_finite(), "Non-finite tensor created"

    def backwards(self) -> list[tuple[Gradient, TrainableWeight]]:
        # Initial gradient is 0s
        grad = Gradient(np.zeros_like(self.data))
        return self.grad_fn.backwards(grad)

    @property
    def shape(self):
        return self.data.shape

    def is_finite(self) -> bool:
        return np.all(np.isfinite(self.data))


@dataclass
class CallableGradFn(LeafGradFn, ABC):
    """
    This grad function will be called invoked when the backward pass is called.


    .backwards() will be called on the last tensor, kicking off the chain of grad_fn calls. This function is responsible for
    1) computing the partials of the output tensor with respect to the input tensor and passing those the the input tensor's grad_fn
    2) computing the partials of the output tensor with respect to the parameters of the module and adding those to the list of weights and their gradients.

    """

    input_tensor: Tensor

    @abstractmethod
    def compute_gradient(
        self, child_grad: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        raise NotImplementedError

    def backwards(self, child_grad: Gradient) -> list[tuple[Gradient, TrainableWeight]]:
        our_input_grad, our_weight_grads = self.compute_gradient(child_grad)

        # Type checking overriden code results to ensure correctness.
        assert isinstance(
            our_input_grad, Gradient
        ), f"Expected Gradient, got {type(our_input_grad)}"
        assert isinstance(
            our_weight_grads, list
        ), f"Expected list, got {type(our_weight_grads)}"
        for grad, weight in our_weight_grads:
            assert isinstance(grad, Gradient), f"Expected Gradient, got {type(grad)}"
            assert isinstance(
                weight, TrainableWeight
            ), f"Expected TrainableWeight, got {type(weight)}"

        parent_weight_grads = self.input_tensor.grad_fn.backwards(our_input_grad)
        merged_weight_grads = our_weight_grads + parent_weight_grads
        return merged_weight_grads


GradFn = CallableGradFn | LeafGradFn
