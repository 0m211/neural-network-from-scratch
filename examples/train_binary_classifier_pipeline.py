import copy
from pathlib import Path

import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU, Sigmoid
from src.nn.sequential import Sequential
from src.nn.losses import binary_cross_entropy
from src.nn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.nn.model_io import save_model, load_model
from src.optim.optimizers import Adam
from src.utils.data_split import train_validation_test_split


def calculate_loss(model, X, y):
    """Calculate loss without updating model parameters."""
    model.eval()

    predictions = model(Tensor(X))

    y = np.asarray(y).reshape(-1,1)

    loss = binary_cross_entropy(
        predictions,
        Tensor(y),
    )

    return float(loss.data)


def predict_probabilities(model, X):
    """Return predicted probabilities in evaluation mode."""
    model.eval()

    predictions = model(Tensor(X))

    return predictions.data.copy()


def main():
    print("=" * 70)
    print("BINARY CLASSIFICATION TRAINING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Generate a reproducible synthetic dataset.
    # --------------------------------------------------

    rng = np.random.default_rng(42)

    X = rng.normal(size=(2000, 2)).astype(np.float64)

    # The model learns whether x1 + x2 > 0.
    y = (
        X[:, 0] + X[:, 1] > 0
    ).astype(np.float64).reshape(-1, 1)

    # Split into training, validation, and test sets.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = train_validation_test_split(
        X,
        y,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        stratify=True,
    )

    print(f"\nTraining samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    # --------------------------------------------------
    # 2. Build the model.
    # --------------------------------------------------

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 1),
        Sigmoid(),
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.01,
    )

    # --------------------------------------------------
    # 3. Configure training.
    # --------------------------------------------------

    epochs = 200
    patience = 15
    min_delta = 1e-5

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_parameters = None
    history = []

    print("\nTraining started...")

    # --------------------------------------------------
    # 4. Train and monitor validation performance.
    # --------------------------------------------------

    for epoch in range(epochs):
        model.train()

        # Forward pass using training data only.
        predictions = model(Tensor(X_train))
        print("Predictions shape:", predictions.data.shape)
        print("Training labels shape:", y_train.shape)

        train_loss_tensor = binary_cross_entropy(
            predictions,
            Tensor(y_train.reshape(-1,1)),
        )

        train_loss = float(train_loss_tensor.data)

        # Backpropagation.
        train_loss_tensor.backward()

        # Update parameters.
        optimizer.step()
        optimizer.zero_grad()

        # Evaluate after updating parameters.
        current_train_loss = calculate_loss(
            model,
            X_train,
            y_train,
        )

        val_loss = calculate_loss(
            model,
            X_val,
            y_val,
        )

        val_probabilities = predict_probabilities(
            model,
            X_val,
        )

        val_predictions = (
            val_probabilities >= 0.5
        ).astype(np.int64)

        val_accuracy = accuracy_score(
            y_val.ravel(),
            val_predictions.ravel(),
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": current_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        # Save the best parameter values in memory.
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_parameters = [
                parameter.data.copy()
                for parameter in model.parameters()
            ]

        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {current_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Accuracy: {val_accuracy:.2%}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1}."
            )
            break

    # --------------------------------------------------
    # 5. Restore the best model parameters.
    # --------------------------------------------------

    if best_parameters is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    for parameter, best_data in zip(
        model.parameters(),
        best_parameters,
    ):
        parameter.data[...] = best_data

    print("\n" + "=" * 70)
    print("BEST CHECKPOINT")
    print("=" * 70)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save the selected model.
    checkpoint_path = Path(
        "artifacts/binary_classifier_best.npz"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model(
        model,
        str(checkpoint_path),
    )

    print(f"Checkpoint saved to: {checkpoint_path}")

    # --------------------------------------------------
    # 6. Evaluate on the test set only now.
    # --------------------------------------------------

    test_probabilities = predict_probabilities(
        model,
        X_test,
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(np.int64)

    actual = y_test.ravel().astype(np.int64)
    predicted = test_predictions.ravel()

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test accuracy:  "
        f"{accuracy_score(actual, predicted):.2%}"
    )

    print(
        f"Test precision: "
        f"{precision_score(actual, predicted):.2%}"
    )

    print(
        f"Test recall:    "
        f"{recall_score(actual, predicted):.2%}"
    )

    print(
        f"Test F1-score:  "
        f"{f1_score(actual, predicted):.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(actual, predicted))

    # --------------------------------------------------
    # 7. Verify checkpoint save/load.
    # --------------------------------------------------

    loaded_model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 1),
        Sigmoid(),
    )

    load_model(
        loaded_model,
        str(checkpoint_path),
    )

    loaded_probabilities = predict_probabilities(
        loaded_model,
        X_test,
    )

    assert np.allclose(
        test_probabilities,
        loaded_probabilities,
        atol=1e-6,
    ), "Loaded model predictions differ from the saved model."

    print("\nPASS: Checkpoint predictions match.")
    print("PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()