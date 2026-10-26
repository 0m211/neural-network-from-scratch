import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

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
    all_predictions = []

    for start in range(0, len(X_test), batch_size):
        end = min(start + batch_size, len(X_test))

        images = X_test[start:end]
        predictions = predict_batch(model, images)

        all_predictions.extend(predictions.tolist())

    return np.array(all_predictions)


def main():
    print("\nLoading MNIST dataset...")

    _, _, X_test, y_test = load_mnist()

    print("\nCreating model...")
    model = create_model()

    print("\nLoading trained weights...")
    load_model(model, "mnist_model.npz")

    print("\nGenerating predictions...")
    predictions = get_predictions(model, X_test)

    accuracy = np.mean(predictions == y_test)

    print("\n" + "=" * 55)
    print("                 MODEL PERFORMANCE")
    print("=" * 55)

    print(f"Total test images: {len(y_test)}")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    print(f"Correct predictions: {np.sum(predictions == y_test)}")
    print(f"Incorrect predictions: {np.sum(predictions != y_test)}")

    print("\nClassification Report:")
    print("-" * 55)

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(10)),
            digits=4,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(10))
    )

    print("\nConfusion Matrix:")
    print(matrix)

    plt.figure(figsize=(10, 8))

    plt.imshow(matrix, interpolation="nearest")
    plt.title("MNIST Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(10))
    plt.yticks(range(10))

    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")

    threshold = matrix.max() / 2

    for row in range(10):
        for col in range(10):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black"
            )

    plt.tight_layout()
    plt.savefig("mnist_confusion_matrix.png", dpi=150)
    plt.show()

    per_digit_accuracy = (
        np.diag(matrix) / matrix.sum(axis=1)
    ) * 100

    print("\nAccuracy for Each Digit:")
    print("-" * 35)

    for digit, digit_accuracy in enumerate(per_digit_accuracy):
        print(f"Digit {digit}: {digit_accuracy:.2f}%")

    errors = np.where(predictions != y_test)[0]

    print("\nFirst 10 Incorrect Predictions:")
    print("-" * 45)

    for index in errors[:10]:
        print(
            f"Image {index}: "
            f"Actual = {y_test[index]}, "
            f"Predicted = {predictions[index]}"
        )

    print("\nConfusion matrix saved to mnist_confusion_matrix.png")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()