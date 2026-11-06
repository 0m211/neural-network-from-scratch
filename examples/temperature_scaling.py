from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from examples.load_mnist import load_mnist
from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential


CHECKPOINT = Path("artifacts/dropout_repeated_best.npz")
OUTPUT_DIR = Path("artifacts")
BATCH_SIZE = 256
SEED = 42


def load_model():
    with np.load(CHECKPOINT) as data:
        dropout = float(data["dropout_probability"].item())
        seed = int(data["seed"].item())
        saved = [data[f"parameter_{i}"].copy() for i in range(4)]

    np.random.seed(seed)

    if dropout == 0:
        model = Sequential(
            Linear(784, 128), ReLU(), Linear(128, 10)
        )
    else:
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Dropout(p=dropout, seed=seed),
            Linear(128, 10),
        )

    for parameter, value in zip(model.parameters(), saved):
        parameter.data[...] = value

    model.eval()
    return model


def get_logits(model, X):
    outputs = []

    for start in range(0, len(X), BATCH_SIZE):
        batch = Tensor(X[start:start + BATCH_SIZE])
        outputs.append(model(batch).data.copy())

    return np.concatenate(outputs)


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy(logits, labels):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    correct_logits = shifted[np.arange(len(labels)), labels]
    return float(np.mean(log_sum_exp - correct_logits))


def find_best_temperature(logits, labels):
    temperatures = np.linspace(0.1, 5.0, 491)

    losses = [
        cross_entropy(logits / temperature, labels)
        for temperature in temperatures
    ]

    best_index = int(np.argmin(losses))

    return float(temperatures[best_index])


def calculate_ece(probabilities, labels, bins=10):
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels

    ece = 0.0

    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins

        mask = (confidence >= lower) & (confidence < upper)

        if i == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)

        if np.any(mask):
            accuracy = np.mean(correct[mask])
            avg_confidence = np.mean(confidence[mask])
            ece += np.mean(mask) * abs(accuracy - avg_confidence)

    return float(ece)


def calculate_brier(probabilities, labels):
    targets = np.eye(10)[labels]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def evaluate(name, logits, labels):
    probabilities = softmax(logits)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = np.mean(predictions == labels)
    loss = cross_entropy(logits, labels)
    ece = calculate_ece(probabilities, labels)
    brier = calculate_brier(probabilities, labels)

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.4%}")
    print(f"Cross-entropy: {loss:.5f}")
    print(f"ECE: {ece:.5f}")
    print(f"Brier score: {brier:.5f}")

    return probabilities


def plot_reliability(before, after, labels):
    plt.figure(figsize=(7, 7))

    for probabilities, name in [
        (before, "Before scaling"),
        (after, "After scaling"),
    ]:
        confidence = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        correct = predictions == labels

        bin_confidence = []
        bin_accuracy = []

        for i in range(10):
            mask = (confidence >= i / 10) & (confidence < (i + 1) / 10)

            if i == 9:
                mask = (confidence >= i / 10) & (confidence <= 1)

            if np.any(mask):
                bin_confidence.append(np.mean(confidence[mask]))
                bin_accuracy.append(np.mean(correct[mask]))

        plt.plot(
            bin_confidence,
            bin_accuracy,
            marker="o",
            label=name,
        )

    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean Confidence")
    plt.ylabel("Accuracy")
    plt.title("Temperature Scaling: Reliability Comparison")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "temperature_scaling_comparison.png"
    plt.savefig(path, dpi=180)
    plt.show()

    print(f"\nSaved: {path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train, y_train, X_test, y_test = load_mnist()

    # Recreate the 54,000 / 6,000 split.
    # This assumes the original experiment used SEED=42.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X_train))

    validation_indices = indices[:6000]

    X_val = X_train[validation_indices]
    y_val = y_train[validation_indices]

    print("Loading model...")
    model = load_model()

    print("Calculating validation and test logits...")
    val_logits = get_logits(model, X_val)
    test_logits = get_logits(model, X_test)

    temperature = find_best_temperature(val_logits, y_val)

    print("\n" + "=" * 50)
    print("TEMPERATURE SCALING RESULTS")
    print("=" * 50)
    print(f"Optimal temperature: {temperature:.3f}")

    before = evaluate("BEFORE SCALING", test_logits, y_test)

    after = evaluate(
        "AFTER SCALING",
        test_logits / temperature,
        y_test,
    )

    plot_reliability(before, after, y_test)

    np.savez(
        OUTPUT_DIR / "temperature_scaling_results.npz",
        temperature=temperature,
        before_probabilities=before,
        after_probabilities=after,
    )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()