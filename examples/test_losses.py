from src.autograd.tensor import Tensor
from src.nn.losses import mse_loss

# Model predictions.
predictions = Tensor([0.8, 0.2, 0.9])

# Correct answers.
targets = Tensor([1.0, 0.0, 1.0])

# Calculate loss.
loss = mse_loss(predictions, targets)

print("MSE Loss:", loss.data)

# Calculate gradients.
loss.backward()

print("Prediction gradients:", predictions.grad)

# Verify expected loss.
assert abs(float(loss.data) - 0.03) < 1e-8

# Expected MSE gradients:
# 2 * (prediction - target) / number_of_examples
expected_gradients = [
    -0.1333333333,
     0.1333333333,
    -0.0666666667
]

assert all(
    abs(float(actual) - expected) < 1e-6
    for actual, expected in zip(
        predictions.grad,
        expected_gradients
    )
)

print("MSE forward and backward tests passed!")