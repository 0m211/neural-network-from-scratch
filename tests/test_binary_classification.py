import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU, Sigmoid
from src.nn.sequential import Sequential
from src.nn.losses import binary_cross_entropy
from src.optim.optimizers import Adam


def main():
    print("=" * 65)
    print("END-TO-END BINARY CLASSIFICATION TEST")
    print("=" * 65)

    # 1. Create a simple dataset.
    # The model must learn whether x1 + x2 > 0.

    np.random.seed(42)

    X = np.random.randn(256, 2).astype(np.float64)

    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    y = y.reshape(-1, 1)

    print(f"\nDataset size: {len(X)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Positive examples: {int(y.sum())}")
    print(f"Negative examples: {len(y) - int(y.sum())}")

    # 2. Build the neural network.

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 1),
        Sigmoid(),
    )

    # 3. Create the optimizer.

    optimizer = Adam(
        model.parameters(),
        lr=0.01,
    )

    # 4. Train the model.

    epochs = 300
    initial_loss = None

    print("\nTraining started...")

    for epoch in range(epochs):

        # Forward propagation.
        predictions = model(Tensor(X))

        # Calculate binary cross-entropy loss.
        loss = binary_cross_entropy(
            predictions,
            Tensor(y),
        )

        if epoch == 0:
            initial_loss = float(loss.data)

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Clear gradients before the next iteration.
        optimizer.zero_grad()

        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {float(loss.data):.6f}"
            )

    # 5. Evaluate the trained model.

    predictions = model(Tensor(X))

    predicted_classes = (
        predictions.data >= 0.5
    ).astype(np.float64)

    accuracy = np.mean(predicted_classes == y) * 100

    final_loss = float(
        binary_cross_entropy(
            predictions,
            Tensor(y),
        ).data
    )

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Accuracy:     {accuracy:.2f}%")

    print("\nSample predictions:")

    for i in range(10):
        print(
            f"Input: {X[i]} | "
            f"Actual: {int(y[i, 0])} | "
            f"Predicted probability: "
            f"{predictions.data[i, 0]:.4f} | "
            f"Predicted class: "
            f"{int(predicted_classes[i, 0])}"
        )

    # 6. Verify that training was successful.

    assert final_loss < initial_loss, (
        "Training failed: loss did not decrease."
    )

    assert accuracy >= 90.0, (
        f"Accuracy is too low: {accuracy:.2f}%"
    )

    print("\nPASS: Loss decreased.")
    print("PASS: Classification accuracy is at least 90%.")
    print("ALL BINARY CLASSIFICATION TESTS PASSED!")


if __name__ == "__main__":
    main()