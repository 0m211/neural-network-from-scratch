
import numpy as np

from src.autograd.tensor import Tensor
from src.nn.losses import binary_cross_entropy


# --------------------------------------------------
# 1. Create predictions and target labels
# --------------------------------------------------

predictions = Tensor([
    [0.3],
    [0.4],
    [0.7],
    [0.8]
])

targets = Tensor([
    [0.0],
    [0.0],
    [1.0],
    [1.0]
])


# --------------------------------------------------
# 2. Display input values
# --------------------------------------------------

print("Predictions:")
print(predictions.data)

print("\nTargets:")
print(targets.data)


# --------------------------------------------------
# 3. Calculate binary cross-entropy loss
# --------------------------------------------------

loss = binary_cross_entropy(predictions, targets)

print("\nLoss before backward:", float(loss.data))


# --------------------------------------------------
# 4. Backpropagation
# --------------------------------------------------

loss.backward()


# --------------------------------------------------
# 5. Inspect prediction gradients
# --------------------------------------------------

print("\nPrediction gradients:")
print(predictions.grad)


# --------------------------------------------------
# 6. Validate gradients
# --------------------------------------------------

gradients_finite = np.all(np.isfinite(predictions.grad))
gradients_nonzero = np.any(
    np.abs(predictions.grad) > 1e-10
)

print("\nGradients finite:", gradients_finite)
print("Gradients nonzero:", gradients_nonzero)

# Every prediction should have a nonzero gradient.
all_gradients_nonzero = np.all(
    np.abs(predictions.grad) > 1e-10
)

print("All gradients nonzero:", all_gradients_nonzero)


# --------------------------------------------------
# 7. Run assertions
# --------------------------------------------------

assert np.isfinite(loss.data).all(), "Loss is not finite."

assert gradients_finite, "Gradients contain NaN or infinity."

assert all_gradients_nonzero, (
    "One or more prediction gradients are zero."
)

print("\nBCE gradient test passed!")

