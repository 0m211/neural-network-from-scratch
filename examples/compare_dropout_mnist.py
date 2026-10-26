import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.dropout import Dropout
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam
from src.nn.model_io import save_model, load_model
from examples.load_mnist import load_mnist


# ==========================================
# 1. CONFIGURATION
# ==========================================

EPOCHS = 10
BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 256
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

SEED = 42

np.random.seed(SEED)


# ==========================================
# 2. LOAD AND SPLIT MNIST
# ==========================================

print("\nLoading MNIST dataset...")

X_train_full, y_train_full, X_test, y_test = load_mnist()

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
# 3. EVALUATION FUNCTION
# ==========================================

def evaluate(model, X, y, batch_size=256):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = len(X)

    for start in range(0, total_samples, batch_size):

        end = min(start + batch_size, total_samples)

        X_batch = X[start:end]
        y_batch = y[start:end]

        logits = model(X_batch)

        loss = cross_entropy(logits, y_batch)

        count = len(X_batch)

        total_loss += float(loss.data) * count

        predictions = np.argmax(logits.data, axis=1)

        total_correct += int(
            np.sum(predictions == y_batch)
        )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ==========================================
# 4. MODEL FACTORY
# ==========================================

def create_model(use_dropout):

    if use_dropout:

        return Sequential(
            Linear(784, 128),
            ReLU(),
            Dropout(p=DROPOUT_RATE, seed=SEED),
            Linear(128, 10)
        )

    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )


# ==========================================
# 5. TRAINING FUNCTION
# ==========================================

def train_model(model, model_name, save_path):

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_epoch = 0

    print(f"\n{'=' * 50}")
    print(f"Training: {model_name}")
    print(f"{'=' * 50}")

    # Use identical data ordering for both experiments.
    rng = np.random.default_rng(SEED)

    training_orders = [
        rng.permutation(len(X_train))
        for _ in range(EPOCHS)
    ]

    for epoch in range(EPOCHS):

        model.train()

        order = training_orders[epoch]

        X_shuffled = X_train[order]
        y_shuffled = y_train[order]

        for start in range(0, len(X_train), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(X_train))

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            logits = model(X_batch)

            loss = cross_entropy(logits, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        # Evaluate after completing the epoch.
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss:       {train_loss:.4f}")
        print(f"Training accuracy:   {train_accuracy * 100:.2f}%")
        print(f"Validation loss:     {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            save_model(model, save_path)

            print("Best validation model saved.")

    # Restore the checkpoint with the lowest validation loss.
    load_model(model, save_path)

    return {
        "name": model_name,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch
    }


# ==========================================
# 6. TRAIN BOTH MODELS
# ==========================================

baseline_results = train_model(
    create_model(use_dropout=False),
    "Baseline Without Dropout",
    "mnist_baseline_comparison.npz"
)

dropout_results = train_model(
    create_model(use_dropout=True),
    "Model With Dropout",
    "mnist_dropout_comparison.npz"
)


# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

print("\n\nFINAL TEST EVALUATION")
print("=" * 50)

results = []

for experiment in [baseline_results, dropout_results]:

    test_loss, test_accuracy = evaluate(
        experiment["model"],
        X_test,
        y_test,
        EVALUATION_BATCH_SIZE
    )

    results.append({
        "name": experiment["name"],
        "best_epoch": experiment["best_epoch"],
        "best_val_loss": experiment["best_val_loss"],
        "val_accuracy": max(experiment["val_accuracies"]),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    print(f"\n{experiment['name']}")
    print("Best epoch:", experiment["best_epoch"])
    print(f"Best validation loss: {experiment['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {max(experiment['val_accuracies']) * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# 8. PLOT TRAINING AND VALIDATION LOSSES
# ==========================================

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    baseline_results["train_losses"],
    marker="o",
    label="Baseline Training Loss"
)

plt.plot(
    epochs,
    baseline_results["val_losses"],
    marker="o",
    label="Baseline Validation Loss"
)

plt.plot(
    epochs,
    dropout_results["train_losses"],
    marker="s",
    label="Dropout Training Loss"
)

plt.plot(
    epochs,
    dropout_results["val_losses"],
    marker="s",
    label="Dropout Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("MNIST: Baseline vs Dropout")

plt.xticks(list(epochs))
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "mnist_dropout_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

print("\nComparison graph saved:")
print("mnist_dropout_comparison.png")


# ==========================================
# 9. SUMMARY
# ==========================================

print("\n\nEXPERIMENT SUMMARY")
print("=" * 50)

for result in results:

    print(
        f"{result['name']}: "
        f"Validation accuracy = {result['val_accuracy'] * 100:.2f}%, "
        f"Test accuracy = {result['test_accuracy'] * 100:.2f}%"
    )

print("\nExperiment completed.")