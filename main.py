import kytorch
import kytorch.nn as nn
import kytorch.optim as optim
import numpy as np

# Set np seed
np.random.seed(42)

# fmt: off
sample_inputs =  [[1, 3, 5],    [1, 9, 5],    [1, 2, 5],    [0, 9, 5],    [0, 0, 5]]
sample_outputs = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# fmt: on

model_layers = [nn.Linear(3, 2), nn.Sin(), nn.Linear(2, 4), nn.Sigmoid()]
model = nn.Sequential(*model_layers)

loss_fn = nn.BCELoss()

opt = optim.GradientDescent(lr=0.1)


for epoch_idx in range(100000):
    epoch_loss = 0
    for sample_input, sample_output in zip(sample_inputs, sample_outputs):
        sample_input = np.array(sample_input, dtype=np.float32)
        sample_output = np.array(sample_output, dtype=np.float32)
        x = kytorch.Tensor(sample_input)
        y = kytorch.Tensor(sample_output)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        weights_to_optimize = loss.backwards()

        opt.step(weights_to_optimize)

        epoch_loss += loss.data.mean()

    if epoch_idx % 100 == 0:
        print(f"Epoch {epoch_idx}: Loss: {epoch_loss / len(sample_inputs)}")
