from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from examples.load_mnist import load_mnist


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEEDS = [42, 123, 2026, 7, 99]

DROPOUT_VALUES = [0.0, 0.2, 0.3]

MAX_EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001

VALIDATION_SIZE = 6000
PATIENCE = 3
MIN_DELTA = 1e-4

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(dropout_probability, seed):
    """
    Create the same network architecture for each experiment.

    Dropout is placed after the ReLU hidden layer.
    """

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

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, batch_size=BATCH_SIZE):
    """
    Evaluate loss and accuracy with dropout disabled.
    """

    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(X)

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)

        X_batch = X[start:end]
        y_batch = y[start:end]

        logits = model(Tensor(X_batch))
        loss = cross_entropy(logits, y_batch)

        batch_size_actual = len(X_batch)

        total_loss += float(loss.data) * batch_size_actual

        predictions = np.argmax(logits.data, axis=1)
        correct_predictions += int(
            np.sum(predictions == y_batch)
        )

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# ============================================================
# TRAIN ONE EXPERIMENT
# ============================================================

def train_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    dropout_probability,
    seed,
):
    """
    Train one model using early stopping.

    The model with the lowest validation loss is restored
    before this function returns.
    """

    print("\n" + "=" * 72)
    print(
        f"SEED={seed} | "
        f"DROPOUT={dropout_probability}"
    )
    print("=" * 72)

    model = create_model(dropout_probability, seed)

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    rng = np.random.default_rng(seed)

    best_val_loss = float("inf")
    best_epoch = 0
    best_parameters = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        shuffled_indices = rng.permutation(len(X_train))

        for start in range(
            0,
            len(shuffled_indices),
            BATCH_SIZE,
        ):
            batch_indices = shuffled_indices[
                start:start + BATCH_SIZE
            ]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            optimizer.zero_grad()

            logits = model(Tensor(X_batch))
            loss = cross_entropy(logits, y_batch)

            loss.backward()
            optimizer.step()

        # Evaluate both datasets with dropout disabled.
        train_loss, train_accuracy = evaluate_model(
            model,
            X_train,
            y_train,
        )

        val_loss, val_accuracy = evaluate_model(
            model,
            X_val,
            y_val,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        # Early stopping checks validation loss.
        if val_loss < best_val_loss - MIN_DELTA:

            best_val_loss = val_loss
            best_epoch = epoch

            best_parameters = [
                parameter.data.copy()
                for parameter in model.parameters()
            ]

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch}."
            )
            break

    # Restore the best validation-loss checkpoint.
    if best_parameters is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    for parameter, best_data in zip(
        model.parameters(),
        best_parameters,
    ):
        parameter.data[...] = best_data

    # Recalculate metrics for the restored checkpoint.
    best_train_loss, best_train_accuracy = evaluate_model(
        model,
        X_train,
        y_train,
    )

    best_val_loss, best_val_accuracy = evaluate_model(
        model,
        X_val,
        y_val,
    )

    print("\nRestored best checkpoint:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Validation accuracy: {best_val_accuracy:.2%}")

    return {
        "seed": seed,
        "dropout": dropout_probability,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_train_loss": best_train_loss,
        "best_train_accuracy": best_train_accuracy,
        "best_parameters": best_parameters,
        "history": history,
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data():
    """
    Load MNIST and create a reproducible training/validation split.
    The official test set is deliberately not returned.
    """

    print("Loading MNIST dataset...")

    X_train_full, y_train_full, X_test, y_test = load_mnist()

    split_rng = np.random.default_rng(42)
    indices = split_rng.permutation(len(X_train_full))

    val_indices = indices[:VALIDATION_SIZE]
    train_indices = indices[VALIDATION_SIZE:]

    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]

    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(
        f"Official test samples reserved: {len(X_test)}"
    )

    # Keep the test data in memory only to emphasize that it
    # is reserved. It is not used for selection or tuning.
    del X_test, y_test

    return X_train, y_train, X_val, y_val


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

def main():
    X_train, y_train, X_val, y_val = prepare_data()

    all_results = []

    for dropout_probability in DROPOUT_VALUES:
        for seed in SEEDS:

            result = train_experiment(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                dropout_probability=dropout_probability,
                seed=seed,
            )

            all_results.append(result)

    # --------------------------------------------------------
    # PRINT INDIVIDUAL RUNS
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("INDIVIDUAL EXPERIMENT RESULTS")
    print("=" * 90)

    print(
        f"{'Dropout':<10}"
        f"{'Seed':<10}"
        f"{'Epoch':<10}"
        f"{'Val Loss':<15}"
        f"{'Val Accuracy':<15}"
    )

    for result in all_results:
        print(
            f"{result['dropout']:<10.2f}"
            f"{result['seed']:<10}"
            f"{result['best_epoch']:<10}"
            f"{result['best_val_loss']:<15.5f}"
            f"{result['best_val_accuracy']:<15.2%}"
        )

    # --------------------------------------------------------
    # CALCULATE MEAN AND STANDARD DEVIATION
    # --------------------------------------------------------

    summaries = []

    print("\n" + "=" * 90)
    print("REPEATED-SEED SUMMARY")
    print("=" * 90)

    print(
        f"{'Dropout':<10}"
        f"{'Mean Val Loss':<20}"
        f"{'Std Val Loss':<18}"
        f"{'Mean Val Acc':<18}"
        f"{'Std Val Acc':<18}"
    )

    for dropout_probability in DROPOUT_VALUES:

        group = [
            result
            for result in all_results
            if result["dropout"] == dropout_probability
        ]

        losses = np.array([
            result["best_val_loss"]
            for result in group
        ])

        accuracies = np.array([
            result["best_val_accuracy"]
            for result in group
        ])

        epochs = np.array([
            result["best_epoch"]
            for result in group
        ])

        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses, ddof=1))

        mean_accuracy = float(np.mean(accuracies))
        std_accuracy = float(np.std(accuracies, ddof=1))

        mean_epoch = float(np.mean(epochs))

        summaries.append({
            "dropout": dropout_probability,
            "mean_val_loss": mean_loss,
            "std_val_loss": std_loss,
            "mean_val_accuracy": mean_accuracy,
            "std_val_accuracy": std_accuracy,
            "mean_best_epoch": mean_epoch,
        })

        print(
            f"{dropout_probability:<10.2f}"
            f"{mean_loss:<20.5f}"
            f"{std_loss:<18.5f}"
            f"{mean_accuracy:<18.2%}"
            f"{std_accuracy:<18.2%}"
        )

    # --------------------------------------------------------
    # SELECT CONFIGURATION BY MEAN VALIDATION LOSS
    # --------------------------------------------------------

    best_summary = min(
        summaries,
        key=lambda item: item["mean_val_loss"],
    )

    selected_dropout = best_summary["dropout"]

    selected_runs = [
        result
        for result in all_results
        if result["dropout"] == selected_dropout
    ]

    # Choose the individual checkpoint with the lowest
    # validation loss among runs of the selected configuration.
    best_run = min(
        selected_runs,
        key=lambda result: result["best_val_loss"],
    )

    print("\n" + "=" * 90)
    print("SELECTED CONFIGURATION")
    print("=" * 90)

    print(f"Dropout probability: {selected_dropout}")
    print(
        "Mean validation loss: "
        f"{best_summary['mean_val_loss']:.5f}"
    )
    print(
        "Mean validation accuracy: "
        f"{best_summary['mean_val_accuracy']:.2%}"
    )
    print(
        "Selected checkpoint seed: "
        f"{best_run['seed']}"
    )
    print(
        "Selected checkpoint validation loss: "
        f"{best_run['best_val_loss']:.5f}"
    )

    # --------------------------------------------------------
    # SAVE EXPERIMENT RESULTS
    # --------------------------------------------------------

    result_file = OUTPUT_DIR / "dropout_repeated_results.npz"

    np.savez(
        result_file,
        dropout_values=np.array(DROPOUT_VALUES),
        seeds=np.array(SEEDS),
        individual_dropout=np.array([
            result["dropout"]
            for result in all_results
        ]),
        individual_seed=np.array([
            result["seed"]
            for result in all_results
        ]),
        individual_best_epoch=np.array([
            result["best_epoch"]
            for result in all_results
        ]),
        individual_val_loss=np.array([
            result["best_val_loss"]
            for result in all_results
        ]),
        individual_val_accuracy=np.array([
            result["best_val_accuracy"]
            for result in all_results
        ]),
        summary_dropout=np.array([
            item["dropout"]
            for item in summaries
        ]),
        summary_mean_val_loss=np.array([
            item["mean_val_loss"]
            for item in summaries
        ]),
        summary_std_val_loss=np.array([
            item["std_val_loss"]
            for item in summaries
        ]),
        summary_mean_val_accuracy=np.array([
            item["mean_val_accuracy"]
            for item in summaries
        ]),
        summary_std_val_accuracy=np.array([
            item["std_val_accuracy"]
            for item in summaries
        ]),
        summary_mean_best_epoch=np.array([
            item["mean_best_epoch"]
            for item in summaries
        ]),
    )

    print(f"\nResults saved to: {result_file}")

    # --------------------------------------------------------
    # SAVE SELECTED MODEL PARAMETERS
    # --------------------------------------------------------

    checkpoint_file = (
        OUTPUT_DIR / "dropout_repeated_best.npz"
    )

    checkpoint_data = {
        f"parameter_{index}": parameter
        for index, parameter in enumerate(
            best_run["best_parameters"]
        )
    }

    checkpoint_data["dropout_probability"] = np.array(
        selected_dropout
    )

    checkpoint_data["seed"] = np.array(
        best_run["seed"]
    )

    checkpoint_data["best_epoch"] = np.array(
        best_run["best_epoch"]
    )

    np.savez(checkpoint_file, **checkpoint_data)

    print(f"Selected parameters saved to: {checkpoint_file}")

    # --------------------------------------------------------
    # PLOT MEAN VALIDATION LOSS
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    dropout_labels = [
        f"Dropout {item['dropout']:.1f}"
        for item in summaries
    ]

    mean_losses = [
        item["mean_val_loss"]
        for item in summaries
    ]

    std_losses = [
        item["std_val_loss"]
        for item in summaries
    ]

    plt.bar(
        dropout_labels,
        mean_losses,
        yerr=std_losses,
        capsize=6,
    )

    plt.title(
        "MNIST Mean Best Validation Loss "
        "Across Five Random Seeds"
    )

    plt.xlabel("Dropout Probability")
    plt.ylabel("Best Validation Cross-Entropy Loss")
    plt.tight_layout()

    plot_file = OUTPUT_DIR / "dropout_repeated_summary.png"

    plt.savefig(plot_file, dpi=150)
    plt.show()

    print(f"Summary plot saved to: {plot_file}")

    # --------------------------------------------------------
    # PLOT MEAN VALIDATION ACCURACY
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    mean_accuracies = [
        item["mean_val_accuracy"] * 100
        for item in summaries
    ]

    std_accuracies = [
        item["std_val_accuracy"] * 100
        for item in summaries
    ]

    plt.bar(
        dropout_labels,
        mean_accuracies,
        yerr=std_accuracies,
        capsize=6,
    )

    plt.title(
        "MNIST Mean Validation Accuracy "
        "Across Five Random Seeds"
    )

    plt.xlabel("Dropout Probability")
    plt.ylabel("Validation Accuracy (%)")
    plt.tight_layout()

    accuracy_plot_file = (
        OUTPUT_DIR / "dropout_repeated_accuracy.png"
    )

    plt.savefig(accuracy_plot_file, dpi=150)
    plt.show()

    print(
        f"Accuracy plot saved to: {accuracy_plot_file}"
    )


if __name__ == "__main__":
    main()