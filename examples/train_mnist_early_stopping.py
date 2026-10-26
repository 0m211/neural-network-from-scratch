import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from src.nn.model_io import save_model, load_model
from examples.load_mnist import load_mnist


# ==========================================
# 1. CONFIGURATION
# ==========================================

MAX_EPOCHS = 20
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 256
LEARNING_RATE = 0.001

PATIENCE = 3
MIN_DELTA = 0.001

np.random.seed(42)


# ==========================================
# 2. LOAD MNIST
# ==========================================

print("\nLoading MNIST dataset...")

X_train_full, y_train_full, X_test, y_test = load_mnist()

print("Dataset loaded successfully.")


# ==========================================
# 3. SPLIT TRAINING AND VALIDATION DATA
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=42,
    stratify=y_train_full
)

print("\nDataset sizes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# 4. CREATE THE MODEL
# ==========================================

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nModel architecture: 784 -> 128 -> 10")
print("Maximum epochs:", MAX_EPOCHS)
print("Early stopping patience:", PATIENCE)
print("Minimum improvement:", MIN_DELTA)


# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================

def evaluate(model, X, y, batch_size=256):

    total_loss = 0.0
    total_correct = 0
    total_samples = len(X)

    for start in range(0, total_samples, batch_size):

        end = min(start + batch_size, total_samples)

        X_batch = X[start:end]
        y_batch = y[start:end]

        logits = model(X_batch)

        loss = cross_entropy(logits, y_batch)

        batch_count = len(X_batch)

        total_loss += float(loss.data) * batch_count

        predictions = np.argmax(logits.data, axis=1)

        total_correct += int(
            np.sum(predictions == y_batch)
        )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==========================================
# 6. EARLY STOPPING VARIABLES
# ==========================================

best_validation_loss = float("inf")

epochs_without_improvement = 0

best_model_path = "mnist_early_stopping_best.npz"

training_losses = []
validation_losses = []
validation_accuracies = []

best_epoch = 0


# ==========================================
# 7. TRAINING LOOP
# ==========================================

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):

    # Shuffle only the training data.
    permutation = np.random.permutation(len(X_train))

    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]

    # Train using mini-batches.
    for start in range(0, len(X_train), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X_train))

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Forward pass.
        logits = model(X_batch)

        # Calculate loss.
        loss = cross_entropy(logits, y_batch)

        # Clear old gradients.
        optimizer.zero_grad()

        # Backpropagation.
        loss.backward()

        # Update parameters.
        optimizer.step()

    # Measure performance after this epoch.
    train_loss, train_accuracy = evaluate(
        model,
        X_train,
        y_train,
        EVALUATION_BATCH_SIZE
    )

    val_loss, val_accuracy = evaluate(
        model,
        X_val,
        y_val,
        EVALUATION_BATCH_SIZE
    )

    training_losses.append(train_loss)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_accuracy)

    print(f"Epoch {epoch}/{MAX_EPOCHS}")
    print(f"Training loss:       {train_loss:.4f}")
    print(f"Training accuracy:   {train_accuracy * 100:.2f}%")
    print(f"Validation loss:     {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

    # --------------------------------------
    # EARLY STOPPING LOGIC
    # --------------------------------------

    if val_loss < best_validation_loss - MIN_DELTA:

        best_validation_loss = val_loss

        best_epoch = epoch

        epochs_without_improvement = 0

        # Save the best model.
        save_model(model, best_model_path)

        print("Validation improved. Best model saved.")

    else:

        epochs_without_improvement += 1

        print(
            "No sufficient validation improvement."
        )

        print(
            "Epochs without improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    print("-" * 45)

    # Stop if patience is exhausted.
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered!")

        print(
            "Validation loss did not improve sufficiently for",
            PATIENCE,
            "consecutive epochs."
        )

        break


# ==========================================
# 8. RESTORE THE BEST MODEL
# ==========================================

print("\nRestoring the best model...")

load_model(model, best_model_path)

print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_validation_loss:.4f}")


# ==========================================
# 9. FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating the restored model on the test set...")

test_loss, test_accuracy = evaluate(
    model,
    X_test,
    y_test,
    EVALUATION_BATCH_SIZE
)

print("\n========== FINAL RESULTS ==========")

print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_validation_loss:.4f}")
print(f"Test loss:            {test_loss:.4f}")
print(f"Test accuracy:        {test_accuracy * 100:.2f}%")

print("===================================")


# ==========================================
# 10. PLOT THE LOSS CURVES
# ==========================================

epochs = range(1, len(training_losses) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    training_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    validation_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")

plt.title("MNIST: Early Stopping Experiment")

plt.xticks(list(epochs))
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "mnist_early_stopping_loss.png",
    dpi=150,
    bbox_inches="tight"
)

print("\nLoss graph saved to:")
print("mnist_early_stopping_loss.png")