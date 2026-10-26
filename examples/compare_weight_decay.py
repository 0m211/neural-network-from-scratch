import os
import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from src.nn.model_io import save_model

from examples.load_mnist import load_mnist


SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SIZE = 6000

WEIGHT_DECAY_VALUES = [0.0, 0.0001, 0.001]

OUTPUT_DIR = "artifacts"
RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "weight_decay_results.npz"
)
PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "weight_decay_comparison.png"
)


def create_model():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )


def evaluate_model(model, X, y, batch_size=256):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = len(y)

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)

        X_batch = Tensor(X[start:end])
        y_batch = y[start:end]

        logits = model(X_batch)
        loss = cross_entropy(logits, y_batch)

        predictions = np.argmax(logits.data, axis=1)

        batch_size_actual = end - start

        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(
            np.sum(predictions == y_batch)
        )

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


def train_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    weight_decay,
    initial_parameters
):
    # Initialize every experiment with identical model parameters.
    model = create_model()

    parameters = model.parameters()

    for parameter, initial_value in zip(
        parameters,
        initial_parameters
    ):
        parameter.data[...] = initial_value

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_parameters = None

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: weight_decay={weight_decay}")
    print("=" * 70)

    for epoch in range(EPOCHS):
        model.train()

        # Use the same shuffled minibatch order for each configuration.
        rng = np.random.default_rng(SEED + epoch)
        indices = rng.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, len(X_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_train))

            X_batch = Tensor(X_shuffled[start:end])
            y_batch = y_shuffled[start:end]

            logits = model(X_batch)
            loss = cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate complete datasets using the updated model.
        train_loss, train_accuracy = evaluate_model(
            model,
            X_train,
            y_train
        )

        val_loss, val_accuracy = evaluate_model(
            model,
            X_val,
            y_val
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_parameters = [
                parameter.data.copy()
                for parameter in model.parameters()
            ]

    # Restore the best validation-loss checkpoint.
    for parameter, best_value in zip(
        model.parameters(),
        best_parameters
    ):
        parameter.data[...] = best_value

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Best validation loss: {best_val_loss:.4f}"
    )

    return {
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": history["val_accuracy"][best_epoch - 1],
        "history": history,
        "model": model
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("MNIST WEIGHT DECAY COMPARISON")
    print("=" * 70)

    print("\nLoading MNIST...")

    X, y, _, _ = load_mnist()

    # Reproduce the split used in the learning-curve experiment.
    split_rng = np.random.default_rng(SEED)
    indices = split_rng.permutation(len(X))

    val_indices = indices[:VALIDATION_SIZE]
    train_indices = indices[VALIDATION_SIZE:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("Official test set: reserved and not loaded for this experiment")

    # Initialize the starting model once so all experiments use
    # exactly the same initial weights.
    np.random.seed(SEED)
    initial_model = create_model()

    initial_parameters = [
        parameter.data.copy()
        for parameter in initial_model.parameters()
    ]

    results = []

    for weight_decay in WEIGHT_DECAY_VALUES:
        result = train_experiment(
            X_train,
            y_train,
            X_val,
            y_val,
            weight_decay,
            initial_parameters
        )

        results.append(result)

    print("\n" + "=" * 70)
    print("WEIGHT DECAY COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"{'Weight Decay':<18}"
        f"{'Best Epoch':<14}"
        f"{'Best Val Loss':<18}"
        f"{'Val Accuracy':<15}"
    )

    for result in results:
        print(
            f"{result['weight_decay']:<18.4g}"
            f"{result['best_epoch']:<14}"
            f"{result['best_val_loss']:<18.4f}"
            f"{result['best_val_accuracy'] * 100:.2f}%"
        )

    best_result = min(
        results,
        key=lambda item: item["best_val_loss"]
    )

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "weight_decay_best.npz"
    )

    save_model(
        best_result["model"],
        best_model_path
    )

    print(
        f"\nBest configuration by validation loss: "
        f"weight_decay={best_result['weight_decay']}"
    )

    print(f"Best model saved to: {best_model_path}")

    # Save results for reproducibility and later comparisons.
    save_data = {}

    for result_index, result in enumerate(results):
        prefix = f"experiment_{result_index}"

        save_data[f"{prefix}_weight_decay"] = np.array(
            result["weight_decay"]
        )

        save_data[f"{prefix}_best_epoch"] = np.array(
            result["best_epoch"]
        )

        save_data[f"{prefix}_best_val_loss"] = np.array(
            result["best_val_loss"]
        )

        save_data[f"{prefix}_best_val_accuracy"] = np.array(
            result["best_val_accuracy"]
        )

        for metric, values in result["history"].items():
            save_data[f"{prefix}_{metric}"] = np.asarray(values)

    np.savez(RESULTS_PATH, **save_data)

    print(f"Experiment results saved to: {RESULTS_PATH}")

    # Compare validation loss across epochs.
    plt.figure(figsize=(10, 6))

    for result in results:
        plt.plot(
            range(1, EPOCHS + 1),
            result["history"]["val_loss"],
            marker="o",
            label=f"Weight decay={result['weight_decay']}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation cross-entropy loss")
    plt.title("MNIST Validation Loss: Weight Decay Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Comparison plot saved to: {PLOT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()