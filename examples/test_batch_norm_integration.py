import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.batch_norm import BatchNorm1d
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam


np.random.seed(42)

print("\nTEST 1: Model Integration")

model = Sequential(
    Linear(4, 8),
    BatchNorm1d(8),
    ReLU(),
    Linear(8, 3)
)

X = np.random.randn(16, 4)
y = np.random.randint(0, 3, size=16)

logits = model(Tensor(X))

print("Input shape:", X.shape)
print("Output shape:", logits.data.shape)

assert logits.data.shape == (16, 3)

print("Model integration passed!")


print("\nTEST 2: Learnable Parameters")

parameters = model.parameters()

print("Total parameter tensors:", len(parameters))

for index, parameter in enumerate(parameters):
    print(f"Parameter {index + 1} shape:", parameter.data.shape)

assert len(parameters) == 6

print("Learnable parameter test passed!")


print("\nTEST 3: Backpropagation")

loss = cross_entropy(logits, y)

for parameter in parameters:
    parameter.grad = np.zeros_like(parameter.data)

loss.backward()

print("Loss:", float(loss.data))

for index, parameter in enumerate(parameters):
    assert parameter.grad.shape == parameter.data.shape
    assert np.all(np.isfinite(parameter.grad))

print("All parameter gradients are finite!")
print("Backpropagation test passed!")


print("\nTEST 4: Optimizer Update")

optimizer = Adam(parameters, lr=0.001)

gamma = model.layers[1].gamma
gamma_before = gamma.data.copy()

optimizer.zero_grad()

logits = model(Tensor(X))
loss = cross_entropy(logits, y)

loss.backward()
optimizer.step()

print("Gamma before update:", gamma_before)
print("Gamma after update:", gamma.data)

assert not np.array_equal(gamma_before, gamma.data)

print("Optimizer update passed!")


print("\nTEST 5: Train and Evaluation Modes")

model.train()

assert model.training
assert model.layers[1].training

model.eval()

assert not model.training
assert not model.layers[1].training

print("Train mode works!")
print("Evaluation mode works!")


print("\nAll Batch Normalization integration tests passed!")