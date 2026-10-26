from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from examples.load_mnist import load_mnist
from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential


CHECKPOINT_PATH = Path("artifacts/dropout_repeated_best.npz")
RESULTS_PATH = Path("artifacts/dropout_test_results.npz")
OUTPUT_DIR = Path("artifacts")

BATCH_SIZE = 256
CONFIDENCE_THRESHOLD = 0.90

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    with np.load(CHECKPOINT_PATH) as checkpoint:
        dropout_probability = float(
            checkpoint["dropout_probability"].item()
        )
        seed = int(checkpoint["seed"].item())

        saved_parameters = [
            checkpoint[f"parameter_{i}"].copy()
            for i in range(4)
        ]

    np.random.seed(seed)

    if dropout_probability == 0.0:
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
        )
    else:
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Dropout(p=dropout_probability, seed=seed),
            Linear(128, 10),
        )

    parameters = model.parameters()

    for parameter, saved_value in zip(
        parameters,
        saved_parameters,
    ):
        if parameter.data.shape != saved_value.shape:
            raise ValueError("Checkpoint parameter shape mismatch.")

        parameter.data[...] = saved_value

    model.eval()

    return model


def load_saved_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(RESULTS_PATH)

    with np.load(RESULTS_PATH) as results:
        predictions = results["predictions"].copy()
        y_test = results["y_test"].copy()

    return predictions, y_test


def get_probabilities(model, X_test):
    batches = []

    model.eval()

    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))

        logits = model(Tensor(X_test[start:end])).data

        # Numerically stable softmax.
        logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        exp_logits = np.exp(logits)

        probabilities = exp_logits / np.sum(
            exp_logits,
            axis=1,
            keepdims=True,
        )

        batches.append(probabilities)

    return np.concatenate(batches, axis=0)


def analyze_confidence(y_test, probabilities):
    predictions = np.argmax(probabilities, axis=1)

    confidence = np.max(probabilities, axis=1)

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    second_highest = sorted_probabilities[:, -2]

    margin = confidence - second_highest

    correct_mask = predictions == y_test
    error_mask = ~correct_mask

    high_confidence_errors = np.flatnonzero(
        error_mask & (confidence >= CONFIDENCE_THRESHOLD)
    )

    print("\n" + "=" * 70)
    print("MODEL CONFIDENCE ANALYSIS")
    print("=" * 70)

    print(f"Test accuracy: {np.mean(correct_mask):.2%}")
    print(f"Total errors: {np.sum(error_mask)}")
    print(
        f"Errors with confidence >= {CONFIDENCE_THRESHOLD:.0%}: "
        f"{len(high_confidence_errors)}"
    )

    print("\nAverage maximum probability:")
    print(f"Correct predictions: {np.mean(confidence[correct_mask]):.4f}")
    print(f"Incorrect predictions: {np.mean(confidence[error_mask]):.4f}")

    print("\nAverage top-two probability margin:")
    print(f"Correct predictions: {np.mean(margin[correct_mask]):.4f}")
    print(f"Incorrect predictions: {np.mean(margin[error_mask]):.4f}")

    print("\nHighest-confidence errors:")
    print(
        f"{'Index':<10}"
        f"{'Actual':<10}"
        f"{'Predicted':<12}"
        f"{'Confidence':<14}"
        f"{'Margin':<10}"
    )

    ranked_errors = high_confidence_errors[
        np.argsort(confidence[high_confidence_errors])[::-1]
    ]

    for index in ranked_errors[:15]:
        print(
            f"{index:<10}"
            f"{int(y_test[index]):<10}"
            f"{int(predictions[index]):<12}"
            f"{confidence[index]:<14.4%}"
            f"{margin[index]:.4f}"
        )

    return predictions, confidence, margin


def plot_confidence_comparison(
    confidence,
    correct_mask,
):
    plt.figure(figsize=(9, 5))

    plt.hist(
        confidence[correct_mask],
        bins=np.linspace(0, 1, 21),
        alpha=0.7,
        label="Correct predictions",
    )

    plt.hist(
        confidence[~correct_mask],
        bins=np.linspace(0, 1, 21),
        alpha=0.7,
        label="Incorrect predictions",
    )

    plt.xlabel("Maximum Softmax Probability")
    plt.ylabel("Number of Images")
    plt.title("Confidence Distribution: Correct vs Incorrect")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "confidence_comparison.png"

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"\nSaved: {path}")


def plot_reliability_diagram(
    confidence,
    correct_mask,
    number_of_bins=10,
):
    edges = np.linspace(0, 1, number_of_bins + 1)

    mean_confidences = []
    accuracies = []
    counts = []

    for bin_index in range(number_of_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]

        if bin_index == number_of_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        mean_confidence = float(np.mean(confidence[mask]))
        accuracy = float(np.mean(correct_mask[mask]))

        mean_confidences.append(mean_confidence)
        accuracies.append(accuracy)
        counts.append(count)

    mean_confidences = np.asarray(mean_confidences)
    accuracies = np.asarray(accuracies)
    counts = np.asarray(counts)

    ece = float(
        np.sum(
            counts / len(confidence)
            * np.abs(mean_confidences - accuracies)
        )
    )

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        mean_confidences,
        accuracies,
        marker="o",
        label="Model",
    )

    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.title(f"MNIST Reliability Diagram (ECE = {ece:.4f})")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "confidence_reliability_diagram.png"

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"\nExpected Calibration Error: {ece:.5f}")
    print(f"Saved: {path}")


def main():
    print("Loading MNIST data...")

    _, _, X_test, dataset_labels = load_mnist()

    saved_predictions, saved_labels = load_saved_results()

    if not np.array_equal(dataset_labels, saved_labels):
        raise ValueError("Saved labels do not match the MNIST test labels.")

    print("Loading selected checkpoint...")

    model = load_checkpoint()

    print("Calculating probabilities...")

    probabilities = get_probabilities(model, X_test)

    predictions, confidence, margin = analyze_confidence(
        saved_labels,
        probabilities,
    )

    if not np.array_equal(predictions, saved_predictions):
        raise ValueError(
            "Loaded model predictions differ from saved predictions."
        )

    correct_mask = predictions == saved_labels

    plot_confidence_comparison(
        confidence,
        correct_mask,
    )

    plot_reliability_diagram(
        confidence,
        correct_mask,
    )

    np.savez(
        OUTPUT_DIR / "confidence_analysis_results.npz",
        probabilities=probabilities,
        predictions=predictions,
        confidence=confidence,
        margin=margin,
        labels=saved_labels,
    )

    print("\nConfidence analysis completed successfully.")


if __name__ == "__main__":
    main()