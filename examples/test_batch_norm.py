import numpy as np

from src.autograd.tensor import Tensor
from src.nn.batch_norm import BatchNorm1d


print("\nTEST 1: Forward Pass")

x = Tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
])

bn = BatchNorm1d(num_features=2)
output = bn(x)

print("Normalized output:")
print(output.data)

print("Output means:", np.mean(output.data, axis=0))
print("Output variances:", np.var(output.data, axis=0))

assert output.data.shape == (4, 2)
assert np.allclose(np.mean(output.data, axis=0), 0.0, atol=1e-6)
assert np.allclose(np.var(output.data, axis=0), 1.0, atol=1e-4)

print("Forward pass passed!")


print("\nTEST 2: Learnable Parameters")

assert len(bn.parameters()) == 2
assert np.allclose(bn.gamma.data, 1.0)
assert np.allclose(bn.beta.data, 0.0)

print("Gamma:", bn.gamma.data)
print("Beta:", bn.beta.data)
print("Parameter test passed!")


print("\nTEST 3: Backward Pass")

bn = BatchNorm1d(num_features=2)

x = Tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
])

output = bn(x)
loss = (output ** 2).mean()

loss.backward()

print("Input gradients:")
print(x.grad)

print("Gamma gradients:")
print(bn.gamma.grad)

print("Beta gradients:")
print(bn.beta.grad)

assert x.grad.shape == x.data.shape
assert bn.gamma.grad.shape == bn.gamma.data.shape
assert bn.beta.grad.shape == bn.beta.data.shape

assert np.all(np.isfinite(x.grad))
assert np.all(np.isfinite(bn.gamma.grad))
assert np.all(np.isfinite(bn.beta.grad))

print("Backward pass passed!")


print("\nTEST 4: Evaluation Mode")

bn.eval()

evaluation_input = Tensor([[2.0, 3.0]])
evaluation_output = bn(evaluation_input)

assert evaluation_output.data.shape == (1, 2)

print("Evaluation output:")
print(evaluation_output.data)
print("Evaluation mode passed!")


print("\nAll Batch Normalization tests passed!")