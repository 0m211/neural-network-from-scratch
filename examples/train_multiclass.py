
import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam


# Reproducible experiment
np.random.seed(42)


# -----------------------------------
# 1. Create a synthetic dataset
# -----------------------------------

X_data = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.2, 0.3],
    [0.3, 0.2],

    [0.8, 0.1],
    [0.9, 0.2],
    [0.7, 0.2],
    [0.8, 0.3],

    [0.1, 0.8],
    [0.2, 0.9],
    [0.2, 0.7],
    [0.3, 0.8]
], dtype=float)


# Class labels: 0, 1, or 2
y_data = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
], dtype=int)


X = Tensor(X_data)


# -----------------------------------
# 2. Build the neural network
# -----------------------------------

model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 3)
)


# -----------------------------------
# 3. Configure the optimizer
# -----------------------------------

optimizer = Adam(
    model.parameters(),
    lr=0.01
)


# -----------------------------------
# 4. Train the model
# -----------------------------------

epochs = 1000

for epoch in range(epochs):

    # Forward pass
    logits = model(X)

    # Calculate loss from raw logits
    loss = cross_entropy(logits, y_data)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch:4d} | "
            f"Loss: {float(loss.data):.6f}"
        )


# -----------------------------------
# 5. Evaluate predictions
# -----------------------------------

logits = model(X)

# Convert logits into probabilities.
# Subtracting the maximum improves numerical stability.
shifted = (
    logits.data
    - np.max(logits.data, axis=1, keepdims=True)
)

exp_values = np.exp(shifted)

probabilities = (
    exp_values
    / np.sum(exp_values, axis=1, keepdims=True)
)

predictions = np.argmax(probabilities, axis=1)

accuracy = np.mean(predictions == y_data) * 100


print("\nFinal predictions:")
for i in range(len(y_data)):
    print(
        f"Input: {X_data[i]} | "
        f"Predicted: {predictions[i]} | "
        f"Actual: {y_data[i]} | "
        f"Probabilities: {np.round(probabilities[i], 3)}"
    )

print(f"\nTraining accuracy: {accuracy:.2f}%")
print("Final loss:", float(cross_entropy(logits, y_data).data))
