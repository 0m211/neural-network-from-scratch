import os
import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.batch_norm import BatchNorm1d
from src.nn.sequential import Sequential
from src.nn.model_io import save_model, load_model


np.random.seed(42)

filepath = "batch_norm_save_load_test.npz"

model = Sequential(
    Linear(4, 8),
    BatchNorm1d(8),
    ReLU(),
    Linear(8, 3)
)

model.train()

X = np.random.randn(32, 4)

for _ in range(5):
    model(Tensor(X))

model.eval()

original_output = model(Tensor(X)).data.copy()
original_mean = model.layers[1].running_mean.copy()
original_var = model.layers[1].running_var.copy()

save_model(model, filepath)

restored_model = Sequential(
    Linear(4, 8),
    BatchNorm1d(8),
    ReLU(),
    Linear(8, 3)
)

load_model(restored_model, filepath)
restored_model.eval()

restored_output = restored_model(Tensor(X)).data

print("\nOriginal running mean:")
print(original_mean)

print("\nRestored running mean:")
print(restored_model.layers[1].running_mean)

print("\nOriginal running variance:")
print(original_var)

print("\nRestored running variance:")
print(restored_model.layers[1].running_var)

assert np.allclose(
    original_mean,
    restored_model.layers[1].running_mean
)

assert np.allclose(
    original_var,
    restored_model.layers[1].running_var
)

assert np.allclose(original_output, restored_output)

print("\nAll Batch Normalization save/load tests passed!")

if os.path.exists(filepath):
    os.remove(filepath)
    print("Temporary test checkpoint removed.")