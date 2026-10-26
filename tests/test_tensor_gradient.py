import numpy as np

from src.autograd.tensor import Tensor


def numerical_gradient(function, values, h=1e-5):
    """
    Calculate numerical gradients element by element
    using central differences.
    """
    values = np.array(values, dtype=float)
    gradients = np.zeros_like(values)

    for index in np.ndindex(values.shape):
        original = values[index]

        values[index] = original + h
        positive = function(values)

        values[index] = original - h
        negative = function(values)

        values[index] = original

        gradients[index] = (positive - negative) / (2 * h)

    return gradients


def function(values):
    x = Tensor(values)

    # A scalar function with a computation graph.
    loss = ((x * x) + 2 * x).sum()

    return float(loss.data)


values = np.array([1.5, -2.0, 3.0])

# Numerical gradients.
numerical = numerical_gradient(function, values)

# Autograd gradients.
x = Tensor(values)
loss = ((x * x) + 2 * x).sum()
loss.backward()

analytical = x.grad

print("Numerical gradients:", numerical)
print("Autograd gradients: ", analytical)

difference = np.max(np.abs(numerical - analytical))

print("Maximum difference:", difference)

assert np.allclose(
    numerical,
    analytical,
    rtol=1e-5,
    atol=1e-6
)

print("Gradient check passed!")