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
NUM_BINS = 15


def softmax(logits):
    shifted_logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exponentials = np.exp(shifted_logits)

    return exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )


def get_predictions(model, X, batch_size=256):
    model.eval()

    all_confidences = []
    all_predictions = []

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

        all_predictions.extend(predictions.tolist())
        all_confidences.extend(confidences.tolist())

    return (
        np.asarray(all_predictions, dtype=np.int64),
        np.asarray(all_confidences, dtype=np.float64)
    )


def calculate_calibration(
    confidences,
    correctness,
    num_bins=15
):
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    total_samples = len(confidences)
    ece = 0.0

    for bin_index in range(num_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count == 0:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            continue

        accuracy = float(np.mean(correctness[mask]))
        confidence = float(np.mean(confidences[mask]))

        bin_accuracies.append(accuracy)
        bin_confidences.append(confidence)

        ece += (
            count / total_samples
        ) * abs(accuracy - confidence)

    return (
        bin_edges,
        np.asarray(bin_accuracies),
        np.asarray(bin_confidences),
        np.asarray(bin_counts),
        ece
    )


def main():
    print("=" * 65)
    print("MNIST MODEL CALIBRATION EVALUATION")
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

    predictions, confidences = get_predictions(
        model,
        X_test,
        BATCH_SIZE
    )

    correctness = predictions == y_test

    (
        bin_edges,
        bin_accuracies,
        bin_confidences,
        bin_counts,
        ece
    ) = calculate_calibration(
        confidences,
        correctness,
        NUM_BINS
    )

    print("\nCalibration by confidence bin:")
    print("-" * 75)
    print(
        f"{'Confidence Range':<20}"
        f"{'Count':>10}"
        f"{'Avg Confidence':>18}"
        f"{'Accuracy':>15}"
    )
    print("-" * 75)

    for i in range(NUM_BINS):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if bin_counts[i] == 0:
            print(
                f"{lower:.2f} - {upper:.2f}"
                f"{0:>10}"
                f"{'N/A':>18}"
                f"{'N/A':>15}"
            )
        else:
            print(
                f"{lower:.2f} - {upper:.2f}"
                f"{bin_counts[i]:>10}"
                f"{bin_confidences[i] * 100:>17.2f}%"
                f"{bin_accuracies[i] * 100:>14.2f}%"
            )

    print("-" * 75)
    print(f"Expected Calibration Error (ECE): {ece:.6f}")
    print(f"Expected Calibration Error: {ece * 100:.2f}%")
    print(f"Test accuracy: {np.mean(correctness) * 100:.2f}%")

    valid_bins = bin_counts > 0

    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        bin_confidences[valid_bins],
        bin_accuracies[valid_bins],
        marker="o",
        label="Model calibration"
    )

    plt.xlabel("Average confidence")
    plt.ylabel("Observed accuracy")
    plt.title("MNIST Calibration Curve")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = "artifacts/calibration_curve.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nCalibration plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()