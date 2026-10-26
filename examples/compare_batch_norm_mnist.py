import os
import numpy as np

from examples.load_mnist import load_mnist
from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.batch_norm import BatchNorm1d
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from src.nn.model_io import save_model, load_model


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PATIENCE = 3

BASELINE_CHECKPOINT = "baseline_mnist_best.npz"
BATCHNORM_CHECKPOINT = "batchnorm_mnist_best.npz"


# ==========================================
# MODEL CREATION
# ==========================================

def create_model(use_batch_norm):
    if use_batch_norm:
        return Sequential(
            Linear(784, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 10)
        )

    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )


# ==========================================
# EVALUATION
# ==========================================

def evaluate(model, X, y):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = len(X)

    for start in range(0, total_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_samples)

        xb = Tensor(X[start:end])
        yb = y[start:end]

        logits = model(xb)
        loss = cross_entropy(logits, yb)

        batch_size = end - start

        total_loss += float(loss.data) * batch_size
        predictions = np.argmax(logits.data, axis=1)
        total_correct += int(np.sum(predictions == yb))

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==========================================
# TRAINING
# ==========================================

def train_model(model_name, use_batch_norm, X_train, y_train, X_val, y_val,
                checkpoint_path):

    print("\n" + "=" * 65)
    print(f"TRAINING: {model_name}")
    print("=" * 65)

    np.random.seed(SEED)

    model = create_model(use_batch_norm)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        indices = np.random.permutation(len(X_train))
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start:start + BATCH_SIZE]

            xb = Tensor(X_train[batch_indices])
            yb = y_train[batch_indices]

            optimizer.zero_grad()

            logits = model(xb)
            loss = cross_entropy(logits, yb)

            loss.backward()
            optimizer.step()

            batch_size = len(batch_indices)

            total_train_loss += float(loss.data) * batch_size
            predictions = np.argmax(logits.data, axis=1)

            total_train_correct += int(np.sum(predictions == yb))
            total_train_samples += batch_size

        train_loss = total_train_loss / total_train_samples
        train_accuracy = total_train_correct / total_train_samples

        val_loss, val_accuracy = evaluate(model, X_val, y_val)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_model(model, checkpoint_path)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Restore the model selected by validation loss.
    load_model(model, checkpoint_path)
    model.eval()

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, best_epoch, best_val_loss


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def main():
    print("Loading MNIST dataset...")

    X, y, X_test, y_test = load_mnist()

    # The project's loader returns the official 60,000/10,000 split.
    # Reserve 6,000 samples from the training partition for validation.
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X))

    val_indices = indices[:6000]
    train_indices = indices[6000:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    baseline_model, baseline_epoch, baseline_val_loss = train_model(
        "Baseline Neural Network",
        False,
        X_train,
        y_train,
        X_val,
        y_val,
        BASELINE_CHECKPOINT
    )

    bn_model, bn_epoch, bn_val_loss = train_model(
        "Neural Network with Batch Normalization",
        True,
        X_train,
        y_train,
        X_val,
        y_val,
        BATCHNORM_CHECKPOINT
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    baseline_test_loss, baseline_test_accuracy = evaluate(
        baseline_model, X_test, y_test
    )

    bn_test_loss, bn_test_accuracy = evaluate(
        bn_model, X_test, y_test
    )

    print("\nBaseline:")
    print(f"  Best epoch:    {baseline_epoch}")
    print(f"  Validation loss: {baseline_val_loss:.4f}")
    print(f"  Test loss:       {baseline_test_loss:.4f}")
    print(f"  Test accuracy:   {baseline_test_accuracy * 100:.2f}%")

    print("\nBatch Normalization:")
    print(f"  Best epoch:    {bn_epoch}")
    print(f"  Validation loss: {bn_val_loss:.4f}")
    print(f"  Test loss:       {bn_test_loss:.4f}")
    print(f"  Test accuracy:   {bn_test_accuracy * 100:.2f}%")

    difference = (
        bn_test_accuracy - baseline_test_accuracy
    ) * 100

    print("\nComparison:")
    print(f"  Accuracy difference: {difference:+.2f} percentage points")

    if difference > 0:
        print("  Batch Normalization performed better on this test run.")
    elif difference < 0:
        print("  The baseline performed better on this test run.")
    else:
        print("  Both models achieved the same test accuracy.")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()