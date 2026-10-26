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


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

VALIDATION_SIZE = 6000

OUTPUT_DIR = "artifacts"
PLOT_PATH = os.path.join(OUTPUT_DIR, "learning_curves.png")
MODEL_PATH = os.path.join(OUTPUT_DIR, "learning_curves_best.npz")


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_loss_and_accuracy(model, X, y, batch_size=256):
    """
    Evaluate average cross-entropy loss and classification accuracy.

    Evaluation does not call backward() or update model parameters.
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

        batch_count = end - start

        total_loss += float(loss.data) * batch_count

        predictions = np.argmax(logits.data, axis=1)
        total_correct += int(np.sum(predictions == y_batch))

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

def train():
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("NEURAL NETWORK LEARNING-CURVE EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Load MNIST
    # --------------------------------------------------------

    print("\nLoading MNIST dataset...")

    X, y, X_test, y_test = load_mnist()

    print(f"Original training samples: {len(X)}")
    print(f"Official test samples:     {len(X_test)}")

    # --------------------------------------------------------
    # 2. Create training and validation sets
    # --------------------------------------------------------

    indices = np.random.permutation(len(X))

    validation_indices = indices[:VALIDATION_SIZE]
    training_indices = indices[VALIDATION_SIZE:]

    X_val = X[validation_indices]
    y_val = y[validation_indices]

    X_train = X[training_indices]
    y_train = y[training_indices]

    print(f"\nTraining samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # --------------------------------------------------------
    # 3. Build the model
    # --------------------------------------------------------

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # 4. Store learning history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_epoch = 0

    # --------------------------------------------------------
    # 5. Train the model
    # --------------------------------------------------------

    for epoch in range(EPOCHS):
        model.train()

        shuffled_indices = np.random.permutation(len(X_train))

        X_shuffled = X_train[shuffled_indices]
        y_shuffled = y_train[shuffled_indices]

        total_training_loss = 0.0
        total_training_correct = 0
        total_training_samples = 0

        for start in range(0, len(X_train), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_train))

            X_batch = Tensor(X_shuffled[start:end])
            y_batch = y_shuffled[start:end]

            # Forward pass.
            logits = model(X_batch)

            # Calculate multiclass cross-entropy.
            loss = cross_entropy(logits, y_batch)

            # Update model parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count = end - start

            total_training_loss += float(loss.data) * batch_count

            predictions = np.argmax(logits.data, axis=1)

            total_training_correct += int(
                np.sum(predictions == y_batch)
            )

            total_training_samples += batch_count

        train_loss = (
            total_training_loss / total_training_samples
        )

        train_accuracy = (
            total_training_correct / total_training_samples
        )

        # ----------------------------------------------------
        # 6. Evaluate validation data
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate_loss_and_accuracy(
            model,
            X_val,
            y_val
        )

        # Record the metrics.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        # Save the best validation checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            save_model(model, MODEL_PATH)

        # Print progress.
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Val Accuracy: {val_accuracy * 100:.2f}%"
        )

    # --------------------------------------------------------
    # 7. Plot learning curves
    # --------------------------------------------------------

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Loss plot.
    plt.subplot(1, 2, 1)

    plt.plot(
        epochs_range,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs_range,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot.
    plt.subplot(1, 2, 2)

    plt.plot(
        epochs_range,
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs_range,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=150)

    print("\n" + "=" * 65)
    print("LEARNING-CURVE EXPERIMENT COMPLETED")
    print("=" * 65)

    print(f"Best epoch:             {best_epoch}")
    print(f"Best validation loss:   {best_val_loss:.4f}")
    print(f"Learning curves saved:  {PLOT_PATH}")
    print(f"Best model saved:       {MODEL_PATH}")

    print("\nOpen the generated plot to inspect your learning curves.")

    plt.show()

    return history


if __name__ == "__main__":
    train()