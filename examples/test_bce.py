import numpy as np

from src.autograd.tensor import Tensor
from src.nn.losses import binary_cross_entropy


predictions = Tensor([0.9, 0.2, 0.8])
targets = Tensor([1.0, 0.0, 1.0])

loss = binary_cross_entropy(predictions, targets)

print("BCE Loss:", float(loss.data))

loss.backward()

print("Gradients:", predictions.grad)

assert np.isfinite(loss.data)
assert np.all(np.isfinite(predictions.grad))

assert float(loss.data) < 0.3

print("BCE forward and backward tests passed!")