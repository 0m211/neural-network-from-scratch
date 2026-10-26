from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from examples.load_mnist import load_mnist


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path("artifacts/dropout_repeated_best.npz")
OUTPUT_DIR = Path("artifacts")
BATCH_SIZE = 256

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

def load_selected_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = np.load(CHECKPOINT_PATH)

    required_keys = {
        "dropout_probability",
        "seed",
        "best_epoch",
        "parameter_0",
        "parameter_1",
        "parameter_2",
        "parameter_3",
    }

    missing_keys = required_keys - set(checkpoint.files)

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing keys: {sorted(missing_keys)}"
        )

    dropout_probability = float(
        checkpoint["dropout_probability"].item()
    )

    seed = int(checkpoint["seed"].item())
    best_epoch = int(checkpoint["best_epoch"].item())

    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError(
            f"Invalid dropout probability: {dropout_probability}"
        )

    # Reconstruct the architecture used during training.
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

    if len(parameters) != 4:
        raise ValueError(
            f"Expected 4 trainable parameters, found {len(parameters)}"
        )

    # Copy saved values into the reconstructed model.
    for index, parameter in enumerate(parameters):
        key = f"parameter_{index}"
        saved_data = checkpoint[key]

        if parameter.data.shape != saved_data.shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"model={parameter.data.shape}, "
                f"checkpoint={saved_data.shape}"
            )

        parameter.data[...] = saved_data

    model.eval()

    print("\nModel checkpoint loaded successfully.")
    print(f"Dropout probability: {dropout_probability}")
    print(f"Checkpoint seed: {seed}")
    print(f"Best validation epoch: {best_epoch}")

    return model


# ============================================================
# LOAD MNIST TEST DATA
# ============================================================

def load_test_data():
    print("\nLoading MNIST dataset...")

    X_train, y_train, X_test, y_test = load_mnist()

    print(f"Official test samples: {len(X_test)}")
    print(f"Input features per sample: {X_test.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_test))}")

    # The training arrays are not needed for this evaluation.
    del X_train, y_train

    return X_test, y_test


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    model.eval()

    total_loss = 0.0
    all_predictions = []

    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))

        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        logits = model(Tensor(X_batch))
        loss = cross_entropy(logits, y_batch)

        batch_count = len(X_batch)
        total_loss += float(loss.data) * batch_count

        predictions = np.argmax(logits.data, axis=1)
        all_predictions.append(predictions)

    predictions = np.concatenate(all_predictions)

    average_loss = total_loss / len(X_test)

    correct = int(np.sum(predictions == y_test))
    incorrect = int(len(y_test) - correct)

    accuracy = correct / len(y_test)

    print("\n" + "=" * 65)
    print("OFFICIAL MNIST TEST RESULTS")
    print("=" * 65)

    print(f"Test samples:       {len(y_test)}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Test accuracy:      {accuracy:.4%}")
    print(f"Test cross-entropy: {average_loss:.5f}")

    return predictions, average_loss, accuracy


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_classification_metrics(y_true, y_pred):
    number_of_classes = 10

    confusion = np.zeros(
        (number_of_classes, number_of_classes),
        dtype=np.int64,
    )

    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1

    print("\n" + "=" * 65)
    print("PER-CLASS CLASSIFICATION METRICS")
    print("=" * 65)

    print(
        f"{'Digit':<10}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'F1-score':<15}"
        f"{'Support':<10}"
    )

    class_results = []

    for digit in range(number_of_classes):
        true_positive = int(confusion[digit, digit])

        false_positive = int(
            confusion[:, digit].sum() - true_positive
        )

        false_negative = int(
            confusion[digit, :].sum() - true_positive
        )

        support = int(confusion[digit, :].sum())

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = precision + recall

        f1_score = (
            2 * precision * recall / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        class_results.append({
            "digit": digit,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        })

        print(
            f"{digit:<10}"
            f"{precision:<15.4f}"
            f"{recall:<15.4f}"
            f"{f1_score:<15.4f}"
            f"{support:<10}"
        )

    macro_precision = float(
        np.mean([item["precision"] for item in class_results])
    )

    macro_recall = float(
        np.mean([item["recall"] for item in class_results])
    )

    macro_f1 = float(
        np.mean([item["f1_score"] for item in class_results])
    )

    print("-" * 65)
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")

    return confusion, class_results


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(confusion):
    plt.figure(figsize=(9, 8))

    plt.imshow(confusion, interpolation="nearest")
    plt.title("MNIST Test Confusion Matrix")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")

    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    plt.colorbar(label="Number of Images")

    threshold = confusion.max() / 2

    for actual in range(10):
        for predicted in range(10):
            value = confusion[actual, predicted]

            plt.text(
                predicted,
                actual,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
                fontsize=8,
            )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "dropout_test_confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"\nConfusion matrix saved to: {output_path}")


# ============================================================
# MOST COMMON MISCLASSIFICATIONS
# ============================================================

def show_common_errors(confusion, limit=10):
    errors = []

    for actual in range(10):
        for predicted in range(10):
            if actual != predicted:
                count = int(confusion[actual, predicted])

                if count > 0:
                    errors.append(
                        (count, actual, predicted)
                    )

    errors.sort(reverse=True)

    print("\n" + "=" * 65)
    print("MOST COMMON MISCLASSIFICATIONS")
    print("=" * 65)

    print(
        f"{'Actual Digit':<18}"
        f"{'Predicted Digit':<20}"
        f"{'Count':<10}"
    )

    for count, actual, predicted in errors[:limit]:
        print(
            f"{actual:<18}"
            f"{predicted:<20}"
            f"{count:<10}"
        )


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

def save_results(
    predictions,
    y_test,
    confusion,
    test_loss,
    test_accuracy,
    class_results,
):
    results_path = OUTPUT_DIR / "dropout_test_results.npz"

    np.savez(
        results_path,
        predictions=predictions,
        y_test=y_test,
        confusion_matrix=confusion,
        test_loss=np.array(test_loss),
        test_accuracy=np.array(test_accuracy),
        class_precision=np.array([
            item["precision"] for item in class_results
        ]),
        class_recall=np.array([
            item["recall"] for item in class_results
        ]),
        class_f1=np.array([
            item["f1_score"] for item in class_results
        ]),
        class_support=np.array([
            item["support"] for item in class_results
        ]),
    )

    print(f"\nNumerical results saved to: {results_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    model = load_selected_model()

    X_test, y_test = load_test_data()

    predictions, test_loss, test_accuracy = evaluate_model(
        model,
        X_test,
        y_test,
    )

    confusion, class_results = calculate_classification_metrics(
        y_test,
        predictions,
    )

    show_common_errors(confusion)

    save_confusion_matrix(confusion)

    save_results(
        predictions=predictions,
        y_test=y_test,
        confusion=confusion,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        class_results=class_results,
    )

    print("\nFinal test evaluation completed successfully.")


if __name__ == "__main__":
    main()