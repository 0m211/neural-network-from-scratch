import numpy as np

from src.autograd.tensor import Tensor


# Two input examples, each containing two features.
X = Tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

# Weight matrix.
W = Tensor([
    [5.0, 6.0],
    [7.0, 8.0]
])

# Bias vector.
b = Tensor([1.0, 2.0])


# Forward pass.
Y = X @ W + b

# Reduce the output to a scalar loss.
loss = Y.sum()

# Backward pass.
loss.backward()


print("Output:")
print(Y.data)

print("\nLoss:")
print(loss.data)

print("\nGradient of X:")
print(X.grad)

print("\nGradient of W:")
print(W.grad)

print("\nGradient of bias:")
print(b.grad)


# Verify the gradients against expected values.
assert np.allclose(
    X.grad,
    [[11.0, 15.0], [11.0, 15.0]]
)

assert np.allclose(
    W.grad,
    [[4.0, 4.0], [6.0, 6.0]]
)

assert np.allclose(
    b.grad,
    [2.0, 2.0]
)

print("\nAll gradient checks passed!")