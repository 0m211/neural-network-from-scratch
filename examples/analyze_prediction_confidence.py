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


def softmax(logits):
    """
    Convert logits into probabilities.

    Subtracting the maximum improves numerical stability.
    """
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exponentials = np.exp(shifted_logits)

    probabilities = exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )

    return probabilities


def get_predictions_and_confidence(
    model,
    X,
    batch_size=256
):
    model.eval()

    all_predictions = []
    all_confidences = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        X_batch = Tensor(X[start:end])

        logits = model(X_batch)

        probabilities = softmax(logits.data)

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        confidences = np.max(
            probabilities,
            axis=1
        )

        all_predictions.extend(
            predictions.tolist()
        )

        all_confidences.extend(
            confidences.tolist()
        )

    return (
        np.array(all_predictions, dtype=np.int64),
        np.array(all_confidences, dtype=np.float64)
    )


def main():
    print("=" * 65)
    print("MNIST PREDICTION CONFIDENCE ANALYSIS")
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

    print("\nCalculating prediction probabilities...")

    predictions, confidences = (
        get_predictions_and_confidence(
            model,
            X_test,
            BATCH_SIZE
        )
    )

    correct_mask = predictions == y_test
    incorrect_mask = ~correct_mask

    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[incorrect_mask]

    accuracy = np.mean(correct_mask)

    print("\n" + "=" * 65)
    print("CONFIDENCE RESULTS")
    print("=" * 65)

    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print(
        "Average confidence on correct predictions: "
        f"{np.mean(correct_confidences) * 100:.2f}%"
    )

    print(
        "Average confidence on incorrect predictions: "
        f"{np.mean(incorrect_confidences) * 100:.2f}%"
    )

    high_confidence_errors = np.where(
        incorrect_mask & (confidences >= 0.90)
    )[0]

    print(
        "\nIncorrect predictions with confidence >= 90%: "
        f"{len(high_confidence_errors)}"
    )

    print("\nFirst 15 high-confidence errors:")

    for index in high_confidence_errors[:15]:
        print(
            f"Index: {index:5d} | "
            f"Actual: {y_test[index]} | "
            f"Predicted: {predictions[index]} | "
            f"Confidence: {confidences[index] * 100:.2f}%"
        )

    plt.figure(figsize=(10, 6))

    plt.hist(
        correct_confidences,
        bins=20,
        alpha=0.6,
        label="Correct predictions"
    )

    plt.hist(
        incorrect_confidences,
        bins=20,
        alpha=0.6,
        label="Incorrect predictions"
    )

    plt.xlabel("Maximum predicted probability")
    plt.ylabel("Number of images")

    plt.title("Prediction Confidence Distribution")

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = "artifacts/confidence_distribution.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"\nConfidence plot saved to: {output_path}"
    )

    plt.show()


if __name__ == "__main__":
    main()