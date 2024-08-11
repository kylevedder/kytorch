from kytorch import TrainableWeight, Gradient
import numpy as np


class GradientDescent:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, inputs: list[tuple[Gradient, TrainableWeight]]):
        for grad, weight in inputs:
            new_value = weight.data - self.lr * grad.data
            assert np.isfinite(
                new_value
            ).all(), f"Non-finite values in weight update: {new_value}"
            weight.update(new_value)
