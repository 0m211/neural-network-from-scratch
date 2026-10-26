import numpy as np

from src.autograd.tensor import Tensor


# Test 1: ReLU
x = Tensor([-2.0, 0.0, 3.0])

y = x.relu().sum()
y.backward()

print("ReLU output:", np.maximum(0, [-2.0, 0.0, 3.0]))
print("ReLU gradient:", x.grad)

assert np.allclose(y.data, 3.0)
assert np.allclose(x.grad, [0.0, 0.0, 1.0])


# Test 2: Sigmoid
x = Tensor([0.0])

y = x.sigmoid().sum()
y.backward()

print("\nSigmoid output:", y.data)
print("Sigmoid gradient:", x.grad)

assert np.allclose(y.data, 0.5)
assert np.allclose(x.grad, [0.25])


# Test 3: Tanh
x = Tensor([0.0])

y = x.tanh().sum()
y.backward()

print("\nTanh output:", y.data)
print("Tanh gradient:", x.grad)

assert np.allclose(y.data, 0.0)
assert np.allclose(x.grad, [1.0])


# Test 4: Mean and power
x = Tensor([1.0, 2.0, 3.0])

y = (x ** 2).mean()
y.backward()

print("\nMean squared value:", y.data)
print("Gradient:", x.grad)

assert np.allclose(y.data, 14.0 / 3.0)
assert np.allclose(x.grad, [2.0 / 3.0, 4.0 / 3.0, 2.0])


print("\nAll tensor operation tests passed!")