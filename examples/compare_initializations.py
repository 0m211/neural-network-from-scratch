import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from examples.load_mnist import load_mnist


def create_model(initialization):
    return Sequential(
        Linear(784, 128, initialization=initialization),
        ReLU(),
        Linear(128, 10, initialization=initialization)
    )


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


def train_one_model(
    initialization,
    X_train,
    y_train,
    X_test,
    y_test,
    training_orders
):
    print("\n" + "=" * 55)
    print(f"TRAINING WITH {initialization.upper()} INITIALIZATION")
    print("=" * 55)

    model = create_model(initialization)

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = len(training_orders)
    batch_size = 64

    for epoch in range(epochs):
        indices = training_orders[epoch]

        shuffled_X = X_train[indices]
        shuffled_y = y_train[indices]

        total_loss = 0.0
        batches = 0

        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))

            images = shuffled_X[start:end]
            labels = shuffled_y[start:end]

            logits = model(Tensor(images))
            loss = cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            batches += 1

        average_loss = total_loss / batches

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training loss: {average_loss:.4f}"
        )

    accuracy = calculate_accuracy(
        model,
        X_test,
        y_test
    )

    print(f"Test accuracy: {accuracy * 100:.2f}%")

    filename = f"mnist_{initialization}_model.npz"

    np.savez(
        filename,
        **{
            f"parameter_{i}": parameter.data
            for i, parameter in enumerate(model.parameters())
        }
    )

    print(f"Model saved to {filename}")

    return accuracy


def main():
    print("Loading MNIST dataset...")

    X_train, y_train, X_test, y_test = load_mnist()

    epochs = 3

    np.random.seed(123)
    training_orders = [
        np.random.permutation(len(X_train))
        for _ in range(epochs)
    ]

    results = {}

    for initialization in ["he", "xavier", "small"]:
        results[initialization] = train_one_model(
            initialization,
            X_train,
            y_train,
            X_test,
            y_test,
            training_orders
        )

    print("\n" + "=" * 55)
    print("INITIALIZATION COMPARISON RESULTS")
    print("=" * 55)

    for initialization, accuracy in results.items():
        print(
            f"{initialization.capitalize():<10} "
            f"Test accuracy: {accuracy * 100:.2f}%"
        )

    best_method = max(results, key=results.get)

    print(f"\nBest result in this run: {best_method.capitalize()}")
    print("Comparison completed successfully!")


if __name__ == "__main__":
    main()