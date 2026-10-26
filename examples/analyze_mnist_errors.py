from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from examples.load_mnist import load_mnist
from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path("artifacts/dropout_repeated_best.npz")
RESULTS_PATH = Path("artifacts/dropout_test_results.npz")
OUTPUT_DIR = Path("artifacts")

BATCH_SIZE = 256
NUM_EXAMPLES = 20
RANDOM_SEED = 42
HIGH_CONFIDENCE_THRESHOLD = 0.90

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD TEST RESULTS
# ============================================================

def load_saved_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    with np.load(RESULTS_PATH) as data:
        predictions = data["predictions"].copy()
        y_test = data["y_test"].copy()

    if predictions.shape != y_test.shape:
        raise ValueError("Predictions and labels have different shapes.")

    print(f"Loaded {len(y_test)} saved test predictions.")

    return predictions, y_test


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    with np.load(CHECKPOINT_PATH) as checkpoint:
        required_keys = {
            "dropout_probability",
            "seed",
            "parameter_0",
            "parameter_1",
            "parameter_2",
            "parameter_3",
        }

        missing = required_keys - set(checkpoint.files)

        if missing:
            raise ValueError(
                f"Missing checkpoint keys: {sorted(missing)}"
            )

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

    if len(parameters) != len(saved_parameters):
        raise ValueError("Checkpoint parameter count does not match model.")

    for parameter, saved_value in zip(parameters, saved_parameters):
        if parameter.data.shape != saved_value.shape:
            raise ValueError(
                f"Parameter shape mismatch: "
                f"{parameter.data.shape} versus {saved_value.shape}"
            )

        parameter.data[...] = saved_value

    model.eval()

    print(f"Loaded model with dropout={dropout_probability}.")
    print(f"Checkpoint seed: {seed}")

    return model


# ============================================================
# LOAD MNIST TEST IMAGES
# ============================================================

def load_test_images():
    _, _, X_test, y_test = load_mnist()

    print(f"Test images shape: {X_test.shape}")

    return X_test, y_test


# ============================================================
# CALCULATE PREDICTIONS AND CONFIDENCE
# ============================================================

def calculate_predictions_and_confidence(model, X_test):
    all_probabilities = []

    model.eval()

    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))

        logits = model(Tensor(X_test[start:end])).data

        # Numerically stable softmax.
        shifted_logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        exponentials = np.exp(shifted_logits)

        probabilities = exponentials / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )

        all_probabilities.append(probabilities)

    probabilities = np.concatenate(all_probabilities, axis=0)

    predictions = np.argmax(probabilities, axis=1)

    confidence = np.max(probabilities, axis=1)

    return predictions, confidence, probabilities


# ============================================================
# PLOT MISCLASSIFIED IMAGES
# ============================================================

def plot_misclassified_examples(
    X_test,
    y_test,
    predictions,
    confidence,
):
    error_indices = np.flatnonzero(predictions != y_test)

    if len(error_indices) == 0:
        print("No misclassified examples found.")
        return

    rng = np.random.default_rng(RANDOM_SEED)

    count = min(NUM_EXAMPLES, len(error_indices))

    selected = rng.choice(
        error_indices,
        size=count,
        replace=False,
    )

    columns = 4
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3.1 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for position, index in enumerate(selected):
        image = X_test[index].reshape(28, 28)

        axes[position].imshow(image, cmap="gray")

        axes[position].set_title(
            f"Actual: {y_test[index]} | Pred: {predictions[index]}\n"
            f"Confidence: {confidence[index]:.1%}",
            fontsize=9,
            pad=8,
        )

        axes[position].axis("off")

    for position in range(count, len(axes)):
        axes[position].axis("off")

    fig.suptitle(
        f"Misclassified MNIST Examples — {count} of "
        f"{len(error_indices)} Errors",
        fontsize=15,
        y=1.01,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "mnist_misclassified_examples.png"

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved misclassified examples: {output_path}")


# ============================================================
# PLOT ERROR RATE BY DIGIT
# ============================================================

def plot_error_rates(y_test, predictions):
    digits = np.arange(10)
    error_rates = []

    print("\nERROR RATE BY ACTUAL DIGIT")
    print("-" * 45)

    for digit in digits:
        mask = y_test == digit
        total = int(np.sum(mask))
        errors = int(np.sum(predictions[mask] != y_test[mask]))

        rate = errors / total if total else 0.0

        error_rates.append(rate)

        print(
            f"Digit {digit}: {errors}/{total} errors "
            f"({rate:.2%})"
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(digits, np.array(error_rates) * 100)

    ax.set_title("MNIST Test Error Rate by Actual Digit")
    ax.set_xlabel("Actual Digit")
    ax.set_ylabel("Error Rate (%)")
    ax.set_xticks(digits)

    ax.bar_label(
        bars,
        fmt="%.2f%%",
        padding=3,
        fontsize=8,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "mnist_error_rate_by_digit.png"

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"Saved error-rate chart: {output_path}")


# ============================================================
# ANALYZE HIGH-CONFIDENCE ERRORS
# ============================================================

def analyze_high_confidence_errors(
    y_test,
    predictions,
    confidence,
):
    error_indices = np.flatnonzero(predictions != y_test)

    high_confidence_errors = error_indices[
        confidence[error_indices] >= HIGH_CONFIDENCE_THRESHOLD
    ]

    print("\n" + "=" * 65)
    print("HIGH-CONFIDENCE MISCLASSIFICATIONS")
    print("=" * 65)

    print(
        f"Confidence threshold: "
        f"{HIGH_CONFIDENCE_THRESHOLD:.0%}"
    )

    print(f"Total errors: {len(error_indices)}")
    print(
        f"Errors with confidence >= {HIGH_CONFIDENCE_THRESHOLD:.0%}: "
        f"{len(high_confidence_errors)}"
    )

    if len(high_confidence_errors) == 0:
        print("No high-confidence errors found.")
        return

    order = np.argsort(
        confidence[high_confidence_errors]
    )[::-1]

    selected = high_confidence_errors[order[:10]]

    print("\nTop high-confidence errors:")

    print(
        f"{'Index':<10}"
        f"{'Actual':<10}"
        f"{'Predicted':<12}"
        f"{'Confidence':<12}"
    )

    for index in selected:
        print(
            f"{index:<10}"
            f"{int(y_test[index]):<10}"
            f"{int(predictions[index]):<12}"
            f"{confidence[index]:.2%}"
        )


# ============================================================
# PLOT CONFIDENCE DISTRIBUTION
# ============================================================

def plot_confidence_distribution(
    y_test,
    predictions,
    confidence,
):
    correct_mask = predictions == y_test
    incorrect_mask = ~correct_mask

    plt.figure(figsize=(9, 5))

    plt.hist(
        confidence[correct_mask],
        bins=20,
        alpha=0.7,
        label="Correct predictions",
    )

    plt.hist(
        confidence[incorrect_mask],
        bins=20,
        alpha=0.7,
        label="Incorrect predictions",
    )

    plt.title("Prediction Confidence: Correct vs Incorrect")
    plt.xlabel("Maximum Predicted Probability")
    plt.ylabel("Number of Images")
    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / "mnist_confidence_distribution.png"

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()

    print(f"Saved confidence distribution: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    saved_predictions, saved_labels = load_saved_results()

    model = load_model()

    X_test, dataset_labels = load_test_images()

    if not np.array_equal(saved_labels, dataset_labels):
        raise ValueError(
            "Saved labels do not match the current MNIST test labels."
        )

    if len(saved_predictions) != len(X_test):
        raise ValueError(
            "Saved predictions and test images have different lengths."
        )

    predictions, confidence, probabilities = (
        calculate_predictions_and_confidence(model, X_test)
    )

    # Verify the reconstructed checkpoint matches the saved results.
    if not np.array_equal(predictions, saved_predictions):
        raise ValueError(
            "Predictions from the loaded checkpoint do not match the "
            "saved test predictions. Check checkpoint and result files."
        )

    accuracy = np.mean(predictions == saved_labels)

    print(f"\nVerified test accuracy: {accuracy:.2%}")

    plot_misclassified_examples(
        X_test,
        saved_labels,
        predictions,
        confidence,
    )

    plot_error_rates(
        saved_labels,
        predictions,
    )

    analyze_high_confidence_errors(
        saved_labels,
        predictions,
        confidence,
    )

    plot_confidence_distribution(
        saved_labels,
        predictions,
        confidence,
    )

    print("\nError analysis completed successfully.")


if __name__ == "__main__":
    main()