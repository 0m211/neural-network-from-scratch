import os
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

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VALIDATION_SIZE = 6000

DROPOUT_VALUES = [0.0, 0.2, 0.3]

OUTPUT_DIR = "artifacts"

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "dropout_comparison_results.npz"
)

PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "dropout_comparison.png"
)


# ============================================================
# MODEL
# ============================================================

def create_model(dropout_probability):
    """
    Create a neural network with configurable dropout.

    Architecture:
        Input: 784
        Hidden: 128 neurons
        ReLU
        Optional Dropout
        Output: 10 classes
    """

    if dropout_probability == 0.0:
        return Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10)
        )

    return Sequential(
        Linear(784, 128),
        ReLU(),
        Dropout(p=dropout_probability, seed=SEED),
        Linear(128, 10)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, batch_size=256):
    """
    Evaluate loss and accuracy.

    Dropout is disabled during evaluation.
    """

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

        predictions = np.argmax(
            logits.data,
            axis=1
        )

        actual_batch_size = end - start

        total_loss += (
            float(loss.data) * actual_batch_size
        )

        total_correct += int(
            np.sum(predictions == y_batch)
        )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

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
    initial_parameters
):
    """
    Train one model and track training/validation metrics.
    """

    model = create_model(dropout_probability)

    # Give every experiment identical starting parameters.
    for parameter, initial_value in zip(
        model.parameters(),
        initial_parameters
    ):
        parameter.data[...] = initial_value

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
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

    print("\n" + "=" * 72)
    print(
        f"EXPERIMENT: dropout_probability="
        f"{dropout_probability}"
    )
    print("=" * 72)

    for epoch in range(EPOCHS):

        model.train()

        # Use the same minibatch order for each configuration.
        rng = np.random.default_rng(SEED + epoch)

        indices = rng.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for start in range(0, len(X_train), BATCH_SIZE):

            end = min(
                start + BATCH_SIZE,
                len(X_train)
            )

            X_batch = Tensor(
                X_shuffled[start:end]
            )

            y_batch = y_shuffled[start:end]

            logits = model(X_batch)

            loss = cross_entropy(
                logits,
                y_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

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

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # SAVE BEST PARAMETERS IN MEMORY
        # ----------------------------------------------------

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
        "dropout_probability": dropout_probability,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": (
            history["val_accuracy"][best_epoch - 1]
        ),
        "history": history,
        "model": model
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 72)
    print("MNIST DROPOUT COMPARISON")
    print("=" * 72)

    print("\nLoading MNIST dataset...")

    X, y, _, _ = load_mnist()

    # Reproduce the same split used in previous experiments.
    split_rng = np.random.default_rng(SEED)

    indices = split_rng.permutation(
        len(X)
    )

    val_indices = indices[:VALIDATION_SIZE]

    train_indices = indices[VALIDATION_SIZE:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(
        "Official test set: reserved for final evaluation"
    )

    # Create the initial model once so all experiments start
    # with exactly the same weights and biases.
    np.random.seed(SEED)

    initial_model = create_model(
        dropout_probability=0.0
    )

    initial_parameters = [
        parameter.data.copy()
        for parameter in initial_model.parameters()
    ]

    results = []

    # --------------------------------------------------------
    # RUN ALL EXPERIMENTS
    # --------------------------------------------------------

    for dropout_probability in DROPOUT_VALUES:

        result = train_experiment(
            X_train,
            y_train,
            X_val,
            y_val,
            dropout_probability,
            initial_parameters
        )

        results.append(result)

    # --------------------------------------------------------
    # PRINT COMPARISON TABLE
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("DROPOUT EXPERIMENT SUMMARY")
    print("=" * 72)

    print(
        f"{'Dropout':<14}"
        f"{'Best Epoch':<14}"
        f"{'Best Val Loss':<18}"
        f"{'Val Accuracy':<16}"
    )

    for result in results:

        print(
            f"{result['dropout_probability']:<14.2f}"
            f"{result['best_epoch']:<14}"
            f"{result['best_val_loss']:<18.4f}"
            f"{result['best_val_accuracy'] * 100:.2f}%"
        )

    # Choose the best configuration using validation loss.
    best_result = min(
        results,
        key=lambda result: result["best_val_loss"]
    )

    print(
        f"\nBest dropout probability: "
        f"{best_result['dropout_probability']}"
    )

    print(
        f"Best validation loss: "
        f"{best_result['best_val_loss']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "dropout_best.npz"
    )

    from src.nn.model_io import save_model

    save_model(
        best_result["model"],
        best_model_path
    )

    print(
        f"Best model saved to: {best_model_path}"
    )

    # --------------------------------------------------------
    # SAVE EXPERIMENT METRICS
    # --------------------------------------------------------

    save_data = {}

    for result_index, result in enumerate(results):

        prefix = f"experiment_{result_index}"

        save_data[
            f"{prefix}_dropout_probability"
        ] = np.array(
            result["dropout_probability"]
        )

        save_data[
            f"{prefix}_best_epoch"
        ] = np.array(
            result["best_epoch"]
        )

        save_data[
            f"{prefix}_best_val_loss"
        ] = np.array(
            result["best_val_loss"]
        )

        save_data[
            f"{prefix}_best_val_accuracy"
        ] = np.array(
            result["best_val_accuracy"]
        )

        for metric, values in result["history"].items():

            save_data[
                f"{prefix}_{metric}"
            ] = np.asarray(values)

    np.savez(
        RESULTS_PATH,
        **save_data
    )

    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # PLOT VALIDATION LOSS
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for result in results:

        plt.plot(
            range(1, EPOCHS + 1),
            result["history"]["val_loss"],
            marker="o",
            label=(
                f"Dropout="
                f"{result['dropout_probability']}"
            )
        )

    plt.xlabel("Epoch")

    plt.ylabel("Validation cross-entropy loss")

    plt.title(
        "MNIST Validation Loss: Dropout Comparison"
    )

    plt.xticks(
        range(1, EPOCHS + 1)
    )

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"Comparison graph saved to: {PLOT_PATH}"
    )

    plt.show()


if __name__ == "__main__":
    main()