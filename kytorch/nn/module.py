from kytorch import Tensor


class Module:

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        res = self.forward(*args, **kwargs)
        assert isinstance(
            res, Tensor
        ), f"Expected forward to return a Tensor, got {res}"
        assert res.is_finite(), f"Got non-finite values in the forward pass of {self}"
        return res


class Sequential(Module):

    def __init__(self, *layers: Module) -> None:
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            assert isinstance(layer, Module), f"Layer {layer} is not a Module"
            x = layer(x)
        return x
