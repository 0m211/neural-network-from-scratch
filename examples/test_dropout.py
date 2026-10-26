import numpy as np

from src.autograd.tensor import Tensor
from src.nn.dropout import Dropout


np.random.seed(42)


# ==========================================
# TEST 1: DROPOUT DURING TRAINING
# ==========================================

print("\nTEST 1: Dropout during training")

x = Tensor(np.ones((1000, 100)))

dropout = Dropout(p=0.2, seed=42)

dropout.train()

output = dropout(x)

dropped_fraction = np.mean(output.data == 0)

print("Input shape:", x.data.shape)
print("Output shape:", output.data.shape)
print("Dropout rate:", dropout.p)
print("Observed dropped fraction:", round(dropped_fraction, 4))

assert output.data.shape == x.data.shape

assert 0.18 < dropped_fraction < 0.22

print("PASS: Dropout is working.")


# ==========================================
# TEST 2: INVERTED DROPOUT SCALING
# ==========================================

print("\nTEST 2: Inverted dropout scaling")

retained_values = output.data[output.data != 0]

assert np.allclose(
    retained_values,
    1.0 / (1.0 - dropout.p)
)

print("Retained activation value:", retained_values[0])

print("PASS: Inverted dropout scaling is correct.")


# ==========================================
# TEST 3: EVALUATION MODE
# ==========================================

print("\nTEST 3: Evaluation mode")

dropout.eval()

evaluation_output = dropout(x)

assert np.array_equal(
    evaluation_output.data,
    x.data
)

assert dropout.mask is None

print("PASS: Dropout is disabled during evaluation.")


# ==========================================
# TEST 4: GRADIENT PROPAGATION
# ==========================================

print("\nTEST 4: Gradient propagation")

dropout.train()

x = Tensor(np.ones((2, 5)))

output = dropout(x)

loss = output.sum()

loss.backward()

expected_gradient = dropout.mask

assert np.allclose(
    x.grad,
    expected_gradient
)

print("Input gradient:")
print(x.grad)

print("PASS: Gradients propagate through dropout.")


# ==========================================
# ALL TESTS COMPLETED
# ==========================================

print("\nAll dropout tests passed!")