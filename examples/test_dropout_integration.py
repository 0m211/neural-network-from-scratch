import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential


np.random.seed(42)


# Create a small neural network.
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Dropout(p=0.5, seed=42),
    Linear(20, 3)
)

x = np.ones((100, 10))


# ==========================================
# TEST 1: TRAINING MODE
# ==========================================

model.train()

dropout_layer = model.layers[2]

assert model.training is True
assert dropout_layer.training is True

print("PASS: Model enters training mode.")


# ==========================================
# TEST 2: EVALUATION MODE
# ==========================================

model.eval()

assert model.training is False
assert dropout_layer.training is False

print("PASS: Model enters evaluation mode.")


# ==========================================
# TEST 3: DROPOUT IS DISABLED IN EVALUATION
# ==========================================

model.eval()

first_output = model(x).data.copy()
second_output = model(x).data.copy()

assert np.allclose(first_output, second_output)

print("PASS: Evaluation predictions are deterministic.")


# ==========================================
# TEST 4: DROPOUT IS ACTIVE DURING TRAINING
# ==========================================

model.train()

first_output = model(x).data.copy()
second_output = model(x).data.copy()

assert not np.allclose(first_output, second_output)

print("PASS: Training dropout generates different outputs.")


# ==========================================
# TEST 5: DROPOUT ADDS NO PARAMETERS
# ==========================================

parameter_count = len(model.parameters())

assert parameter_count == 4

print("PASS: Only trainable weights and biases are collected.")


# ==========================================
# TEST 6: BACKPROPAGATION
# ==========================================

model.train()

output = model(x)

loss = output.sum()

loss.backward()

for parameter in model.parameters():
    assert parameter.grad is not None

print("PASS: Gradients propagate through the full model.")

print("\nAll integration tests passed!")