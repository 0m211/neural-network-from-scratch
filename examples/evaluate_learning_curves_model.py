import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.model_io import load_model

from examples.load_mnist import load_mnist


MODEL_PATH = "artifacts/learning_curves_best.npz"
BATCH_SIZE = 256


def get_predictions(model, X, batch_size=256):
    model.eval()

    predictions = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        X_batch = Tensor(X[start:end])
        logits = model(X_batch)

        batch_predictions = np.argmax(
            logits.data,
            axis=1
        )

        predictions.extend(batch_predictions.tolist())

    return np.array(predictions, dtype=np.int64)


def calculate_confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1

    return matrix


def main():
    print("=" * 65)
    print("MNIST CONFUSION MATRIX ANALYSIS")
    print("=" * 65)

    print("\nLoading MNIST dataset...")

    _, _, X_test, y_test = load_mnist()

    print("\nRebuilding model...")

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )

    print("\nLoading best checkpoint...")

    load_model(model, MODEL_PATH)

    print("\nGenerating predictions...")

    y_pred = get_predictions(
        model,
        X_test,
        BATCH_SIZE
    )

    matrix = calculate_confusion_matrix(
        y_test,
        y_pred
    )

    accuracy = np.mean(y_true_equal := (y_test == y_pred))

    print("\nConfusion Matrix:")
    print("Rows = Actual digits")
    print("Columns = Predicted digits")
    print()

    print(matrix)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nIncorrect predictions by actual digit:")

    for digit in range(10):
        total = np.sum(y_test == digit)
        correct = matrix[digit, digit]
        incorrect = total - correct

        print(
            f"Digit {digit}: "
            f"{incorrect} incorrect out of {total}"
        )

    print("\nMost frequent confusion pairs:")

    confusion_pairs = []

    for actual in range(10):
        for predicted in range(10):
            if actual != predicted:
                count = matrix[actual, predicted]

                if count > 0:
                    confusion_pairs.append(
                        (count, actual, predicted)
                    )

    confusion_pairs.sort(reverse=True)

    for count, actual, predicted in confusion_pairs[:10]:
        print(
            f"Actual {actual} predicted as {predicted}: "
            f"{count} images"
        )

    plt.figure(figsize=(10, 8))

    plt.imshow(matrix, interpolation="nearest")
    plt.title("MNIST Confusion Matrix")
    plt.colorbar(label="Number of Images")

    plt.xticks(range(10))
    plt.yticks(range(10))

    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")

    threshold = matrix.max() / 2

    for row in range(10):
        for col in range(10):
            value = matrix[row, col]

            plt.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black"
            )

    plt.tight_layout()

    output_path = "artifacts/confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nConfusion matrix saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()