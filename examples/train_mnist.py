 
import numpy as np

from src.nn.model_io import save_model
from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam

from examples.load_mnist import load_mnist


def accuracy(model, X, y, batch_size=256):
    """Evaluate classification accuracy in batches."""

    correct = 0
    total = len(y)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        X_batch = Tensor(X[start:end])
        logits = model(X_batch)

        predictions = np.argmax(logits.data, axis=1)

        correct += np.sum(predictions == y[start:end])

    return correct / total


def train():
    np.random.seed(42)

    # 1. Load data
    X_train, y_train, X_test, y_test = load_mnist()

    # 2. Create the model
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )

    # 3. Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    # 4. Training configuration
    epochs = 3
    batch_size = 64

    for epoch in range(epochs):
        # Shuffle the training examples each epoch.
        indices = np.random.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))

            X_batch = Tensor(X_shuffled[start:end])
            y_batch = y_shuffled[start:end]

            # Forward pass
            logits = model(X_batch)

            # Calculate loss from raw logits
            loss = cross_entropy(logits, y_batch)

            # Backpropagation and parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            num_batches += 1

        average_loss = total_loss / num_batches

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Average loss: {average_loss:.4f}"
        )

        # Evaluate on a subset of the test set during development.
        # The full test set is reserved for final evaluation.
        subset_size = 2000

        subset_accuracy = accuracy(
            model,
            X_test[:subset_size],
            y_test[:subset_size]
        )

        print(
            f"First {subset_size} test images | "
            f"Accuracy: {subset_accuracy * 100:.2f}%"
        )

    # Final evaluation on the complete test set.
    test_accuracy = accuracy(model, X_test, y_test)

    print("\nTraining completed!")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")

    save_model(model, "mnist_model.npz")

    return model


if __name__ == "__main__":
    train()

