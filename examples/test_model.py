import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential


np.random.seed(42)

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Simulate a batch of 32 images.
X = Tensor(np.random.randn(32, 784))

# Forward pass.
output = model(X)

print("Input shape:", X.data.shape)
print("Output shape:", output.data.shape)
print("Number of parameters:", sum(
    p.data.size for p in model.parameters()
))

# Check dimensions.
assert output.data.shape == (32, 10)

# Create a simple scalar objective for testing gradients.
loss = (output * output).mean()

loss.backward()

# Verify that trainable parameters receive gradients.
for parameter in model.parameters():
    assert parameter.grad.shape == parameter.data.shape
    assert np.all(np.isfinite(parameter.grad))

print("Forward pass successful!")
print("Backward pass successful!")