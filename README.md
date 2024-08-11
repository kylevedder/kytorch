# KyTorch

A numpy only deep learning framework with a PyTorch-like API.

This project was designed to practice building an autodiff library from scratch. It is not intended to be used in production.

## Autodiff

This is a simple [reverse-mode](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_and_reverse_accumulation) autodiff implementation, built around the copy-on-edit `Tensor` class, where each `Tensor` has data as well as a `GradFn` describing the gradient for the various operations that have been accumulated.

Input `Tensor`s are created with a `LeafGradFn`, which is an empty `GradFn` that terminates the recursion of the computation graph.

### Forwards pass

During the forward pass, every edit operation creates a new `Tensor` with a `GradFn` that has a reference to the pre-operation `Tensor` (and thus its `Tensor`'s `GradFn`); the new `GradFn` describes the derivative of the edit operation done. Importantly, we assume that while an operation may consume multiple tensors, it only produces one tensor as output -- there is only a single output derivative in the computation tree.

### Backwards pass

During the backwards pass, the `backward` method is called on the final `Tensor` in the computation graph (i.e. the loss tensor), initialized with a `0` gradient. The loss gradient will ignore this input gradient and compute a gradient with respect to the loss, kicking off recursive calls to `backward` on the `GradFn` of the `Tensor`s that use the call stack to perform depth first search of the computation graph.

Each call passes in the output's gradient with respect to the loss (i.e. the last operation) so far. The `GradFn` then computes the gradient for the input tensor(s) and the trainable weights with respect to the output tensor, and then combines these to get gradients with respect to the loss (calculus' chain rule). This is reflected in the type signature of `CallableGradFn`'s `compute_gradient()` method, which assumes one input tensor (the first element of the return tuple):

```python
def compute_gradient(
        self, loss_wrt_output: Gradient
    ) -> tuple[Gradient, list[tuple[Gradient, TrainableWeight]]]:
        ...
```

Because this `GradFn` contains a reference to the input `Tensor` (or `Tensor`s), it can then call their respective `backward` methods using the computed gradient with respect to the loss. It then returns the (length 0 or more) list of `TrainableWeight`s and their gradients, which will be given to the optimizer to update their values. For fixed operations like `Sigmoid`, this list will be empty, but for trainable operations like `Linear`, it will contain the weights and their gradients.

## Optimization

Unlike PyTorch, the completed backwards pass directly returns a list of `TrainableWeight`s and their gradients(`list[tuple[Gradient, TrainableWeight]]`). Optimization can then be done by updating the weights using their gradients; for simple Gradient Descent, this is just subtracting the gradient scaled by a learning rate from the weight.

## Demos

To run the demos, you have to add the library to your Python path. As an example:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python demos/two_layer_binary_classification.py 
```

should produce

```
Epoch 0: Loss: 0.7493288099765778
Epoch 100: Loss: 0.1466349795460701
Epoch 200: Loss: 0.18007928729057313
...
Epoch 1700: Loss: 0.0010693292832002043
Epoch 1800: Loss: 0.0010082000051625073
Epoch 1900: Loss: 0.0009537238045595586
```
