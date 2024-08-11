from kytorch import Tensor, CallableGradFn, TrainableWeight, Gradient, GradFn
from dataclasses import dataclass
from .module import Module

import numpy as np


@dataclass
class BinaryCrossEntropyLossGradFn(CallableGradFn):
    y_tensor: Tensor

    def compute_gradient(
        self, loss_wrt_child: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        yhat = self.input_tensor.data
        y = self.y_tensor.data
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        loss_wrt_yhat = (yhat - y) / (yhat - yhat**2)
        loss_wrt_yhat = Gradient(loss_wrt_yhat)
        return loss_wrt_yhat, []


class BinaryCrossEntropyLoss(Module):
    """
    Equivalent to torch.nn.BCELoss(reduction="sum")
    """

    def forward(self, yhat_tensor: Tensor, y_tensor: Tensor):
        yhat = yhat_tensor.data
        y = y_tensor.data
        assert yhat.shape == y.shape, f"Shape difference: {yhat} vs {y}"
        assert np.all(1 >= yhat) and np.all(yhat >= 0), f"Domain error for yhat: {yhat}"
        assert np.all(1 >= y) and np.all(y >= 0), f"Domain error for y: {y}"

        # Ensure all the values are finite
        assert np.all(np.isfinite(yhat)), f"Non-finite values in yhat: {yhat}"
        assert np.all(np.isfinite(y)), f"Non-finite values in y: {y}"

        # Clamp yhat to avoid log(0) and log(1)
        yhat = np.clip(yhat, 1e-7, 1 - 1e-7)
        # We drop the 1/len(yhat) factor to make the loss the same as torch's BCE loss
        loss = -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

        assert np.all(
            np.isfinite(loss)
        ), f"Non-finite values in loss: {loss} from yhat: {yhat} and y: {y}"

        # clamp the loss entries to at most 100 to avoid nan (and like torch's BCE loss)
        loss = np.clip(loss, -100, 100)
        return Tensor(loss, BinaryCrossEntropyLossGradFn(yhat_tensor, y_tensor))
