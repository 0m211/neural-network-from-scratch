import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from examples.load_mnist import load_mnist


def create_model():
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )
    return model


def calculate_accuracy(model, X, y, batch_size=256):
    correct = 0

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        images = X[start:end]
        labels = y[start:end]

        logits = model(Tensor(images)).data
        predictions = np.argmax(logits, axis=1)

        correct += np.sum(predictions == labels)

    return correct / len(y)


def train_model():
    print("\nLoading MNIST dataset...")

    X_train, y_train, X_test, y_test = load_mnist()

    print("\nCreating deeper neural network...")

    model = create_model()

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 3
    batch_size = 64

    print("\nStarting training...")
    print("Architecture: 784 -> 128 -> 64 -> 10")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))

        shuffled_X = X_train[indices]
        shuffled_y = y_train[indices]

        total_loss = 0.0
        number_of_batches = 0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))

            images = shuffled_X[start:end]
            labels = shuffled_y[start:end]

            inputs = Tensor(images)
            logits = model(inputs)

            loss = cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            number_of_batches += 1

        average_loss = total_loss / number_of_batches

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Average loss: {average_loss:.4f}"
        )

    print("\nTraining completed!")

    print("\nEvaluating deeper model on test data...")

    test_accuracy = calculate_accuracy(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 45)
    print("       DEEPER MODEL RESULTS")
    print("=" * 45)

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    np.savez(
        "mnist_deeper_model.npz",
        **{
            f"parameter_{i}": parameter.data
            for i, parameter in enumerate(model.parameters())
        }
    )

    print("\nModel saved to mnist_deeper_model.npz")

    return test_accuracy


if __name__ == "__main__":
    train_model()