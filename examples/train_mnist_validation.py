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

EPOCHS = 3
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 256
LEARNING_RATE = 0.001

np.random.seed(42)


# ==========================================
# 2. LOAD MNIST DATASET
# ==========================================

print("\nLoading MNIST dataset...")

X_train_full, y_train_full, X_test, y_test = load_mnist()

print("Dataset loaded successfully.")

print("Original training images:", X_train_full.shape)
print("Original test images:", X_test.shape)


# ==========================================
# 3. CREATE TRAINING AND VALIDATION SETS
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,
    random_state=42,
    stratify=y_train_full
)

print("\nDataset split completed.")

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Test images:", X_test.shape)


# ==========================================
# 4. BUILD THE NEURAL NETWORK
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

print("\nNeural network created.")
print("Architecture: 784 -> 128 -> 10")
print("Optimizer: Adam")
print("Learning rate:", LEARNING_RATE)


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

        batch_size_actual = len(X_batch)

        total_loss += float(loss.data) * batch_size_actual

        predictions = np.argmax(logits.data, axis=1)

        total_correct += int(
            np.sum(predictions == y_batch)
        )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==========================================
# 6. TRAIN THE MODEL
# ==========================================

training_losses = []
validation_losses = []
validation_accuracies = []

best_validation_loss = float("inf")
best_model_path = "mnist_validation_best.npz"

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # Shuffle training data for this epoch.
    permutation = np.random.permutation(len(X_train))

    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    # Train using mini-batches.
    for start in range(0, len(X_train), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X_train))

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        # Forward pass.
        logits = model(X_batch)

        # Calculate loss.
        loss = cross_entropy(logits, y_batch)

        # Clear gradients from the previous update.
        optimizer.zero_grad()

        # Backpropagation.
        loss.backward()

        # Update weights and biases.
        optimizer.step()

    # Evaluate after completing this epoch.
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

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss:     {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation loss:   {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

    # Save the model whenever validation loss improves.
    if val_loss < best_validation_loss:

        best_validation_loss = val_loss

        save_model(model, best_model_path)

        print("New best model saved!")

    print("-" * 45)


# ==========================================
# 7. LOAD THE BEST MODEL
# ==========================================

print("\nLoading the best model...")

load_model(model, best_model_path)


# ==========================================
# 8. FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating on the untouched test dataset...")

test_loss, test_accuracy = evaluate(
    model,
    X_test,
    y_test,
    EVALUATION_BATCH_SIZE
)

print("\n========== FINAL RESULTS ==========")

print(f"Best validation loss: {best_validation_loss:.4f}")
print(f"Test loss:            {test_loss:.4f}")
print(f"Test accuracy:        {test_accuracy * 100:.2f}%")

print("===================================")


# ==========================================
# 9. PLOT TRAINING AND VALIDATION LOSS
# ==========================================

epochs = range(1, EPOCHS + 1)

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
plt.title("MNIST: Training vs Validation Loss")

plt.xticks(list(epochs))
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "mnist_training_validation_loss.png",
    dpi=150,
    bbox_inches="tight"
)

print("\nLoss graph saved successfully:")
print("mnist_training_validation_loss.png")
print("Open this image from your project folder to view it.")