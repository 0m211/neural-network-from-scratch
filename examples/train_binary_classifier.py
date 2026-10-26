import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU, Sigmoid
from src.nn.sequential import Sequential
from src.nn.losses import binary_cross_entropy
from src.optim.optimizers import Adam


# Make results reproducible.
np.random.seed(42)


# --------------------------------------------------
# 1. Create a synthetic dataset
# --------------------------------------------------

X_data = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.2, 0.3],
    [0.3, 0.2],
    [0.7, 0.8],
    [0.8, 0.7],
    [0.9, 0.8],
    [0.8, 0.9],
], dtype=float)

y_data = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
], dtype=float)


X = Tensor(X_data)
y = Tensor(y_data)


# --------------------------------------------------
# 2. Build the neural network
# --------------------------------------------------

model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)


# --------------------------------------------------
# 3. Configure the optimizer
# --------------------------------------------------

optimizer = Adam(
    model.parameters(),
    lr=0.01
)


# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

epochs = 1000

for epoch in range(epochs):

    # Clear old gradients if the optimizer supports it.
    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()

    # Forward pass.
    predictions = model(X)

    # Calculate binary cross-entropy loss.
    loss = binary_cross_entropy(predictions, y)

    # Backpropagation.
    loss.backward()

    # First-epoch gradient and weight-update diagnostics.
    if epoch == 0:
        print("\n--- Gradient Diagnostics ---")

        parameters = model.parameters()

        for i, parameter in enumerate(parameters):
            print(
                f"Parameter {i}: "
                f"gradient norm = "
                f"{np.linalg.norm(parameter.grad):.10f}"
            )

        # Save the first parameter before the update.
        first_weight = parameters[0]
        weight_before = first_weight.data.copy()

        # Update parameters exactly once.
        optimizer.step()

        # Save the parameter after the update.
        weight_after = first_weight.data.copy()

        # Check whether the parameter changed.
        print(
            "\nWeight changed:",
            not np.allclose(weight_before, weight_after)
        )

    else:
        # Update parameters once in every subsequent epoch.
        optimizer.step()

    # Print training progress.
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch:4d} | "
            f"Loss: {float(loss.data):.6f}"
        )


# --------------------------------------------------
# 5. Evaluate the model
# --------------------------------------------------

print("\nFinal predictions:")

# Calculate predictions after training.
predictions = model(X)

# Calculate final loss.
final_loss = binary_cross_entropy(predictions, y)

# Initialize the correct-prediction counter.
correct = 0

for i in range(len(X_data)):

    probability = float(predictions.data[i, 0])

    predicted_class = int(probability >= 0.5)
    actual_class = int(y_data[i, 0])

    if predicted_class == actual_class:
        correct += 1

    print(
        f"Input: {X_data[i]} | "
        f"Probability: {probability:.4f} | "
        f"Predicted: {predicted_class} | "
        f"Actual: {actual_class}"
    )


# --------------------------------------------------
# 6. Display final results
# --------------------------------------------------

accuracy = correct / len(X_data) * 100

print(f"\nFinal loss: {float(final_loss.data):.6f}")
print(f"Correct predictions: {correct}/{len(X_data)}")
print(f"Training accuracy: {accuracy:.2f}%")

