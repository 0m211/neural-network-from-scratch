import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.model_io import load_model
from examples.load_mnist import load_mnist


def create_model():
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )
    return model


def predict_batch(model, images):
    logits = model(Tensor(images)).data
    predictions = np.argmax(logits, axis=1)
    return predictions


def get_predictions(model, X_test, batch_size=256):
    predictions = []

    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))

        batch = X_test[start:end]
        batch_predictions = predict_batch(model, batch)

        predictions.extend(batch_predictions.tolist())

    return np.array(predictions)


def main():
    print("\nLoading MNIST dataset...")

    _, _, X_test, y_test = load_mnist()

    print("\nCreating model...")
    model = create_model()

    print("\nLoading trained model...")
    load_model(model, "mnist_model.npz")

    print("\nGenerating predictions...")
    predictions = get_predictions(model, X_test)

    incorrect_indices = np.where(predictions != y_test)[0]

    print(f"\nTotal test images: {len(y_test)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}")

    if len(incorrect_indices) == 0:
        print("No incorrect predictions found!")
        return

    number_to_show = min(25, len(incorrect_indices))
    selected_indices = incorrect_indices[:number_to_show]

    columns = 5
    rows = (number_to_show + columns - 1) // columns

    plt.figure(figsize=(12, 3 * rows))

    for position, image_index in enumerate(selected_indices):
        image = X_test[image_index].reshape(28, 28)
        actual = int(y_test[image_index])
        predicted = int(predictions[image_index])

        plt.subplot(rows, columns, position + 1)
        plt.imshow(image, cmap="gray")
        plt.title(
            f"Actual: {actual} | Predicted: {predicted}"
        )
        plt.axis("off")

    plt.suptitle(
        "MNIST: Incorrect Predictions",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig("mnist_misclassified_images.png", dpi=150)
    plt.show()

    print("\nSaved image grid to mnist_misclassified_images.png")

    print("\nIncorrect predictions by actual digit:")

    for digit in range(10):
        count = np.sum(y_test[incorrect_indices] == digit)
        print(f"Digit {digit}: {count} mistakes")


if __name__ == "__main__":
    main()