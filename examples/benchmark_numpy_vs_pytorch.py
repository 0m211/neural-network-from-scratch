import os
import time
import json
import numpy as np
import torch
import torch.nn as nn

from threadpoolctl import threadpool_limits, threadpool_info

from src.autograd.tensor import Tensor, no_grad
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.losses import cross_entropy
from src.optim.optimizers import Adam

from examples.load_mnist import load_mnist


# ==========================================
# CONFIGURATION
# ==========================================

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

SEED = 42
TRIALS = 3

INFERENCE_REPEATS = 100
INFERENCE_SAMPLES = 1000

DEVICE = torch.device("cpu")


# ==========================================
# CPU THREAD CONFIGURATION
# ==========================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

threadpool_limits(limits=1)

print("\nCPU THREAD CONFIGURATION")
print("-" * 40)

for pool in threadpool_info():
    print(
        f"{pool['internal_api']}: "
        f"{pool['num_threads']} thread(s)"
    )


# ==========================================
# PYTORCH MODEL
# ==========================================

class PyTorchNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# MODEL CREATION
# ==========================================

def create_models(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # NumPy model

    numpy_model = Sequential(
        Linear(784, 128, initialization="he"),
        ReLU(),
        Linear(128, 10, initialization="he")
    )

    numpy_optimizer = Adam(
        numpy_model.parameters(),
        lr=LEARNING_RATE
    )

    # PyTorch model

    torch_model = PyTorchNetwork().to(DEVICE)

    torch_optimizer = torch.optim.Adam(
        torch_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    # Copy identical initial weights

    with torch.no_grad():

        torch_model.network[0].weight.copy_(
            torch.from_numpy(
                numpy_model.layers[0].weight.data.T.copy()
            )
        )

        torch_model.network[0].bias.copy_(
            torch.from_numpy(
                numpy_model.layers[0].bias.data.copy()
            )
        )

        torch_model.network[2].weight.copy_(
            torch.from_numpy(
                numpy_model.layers[2].weight.data.T.copy()
            )
        )

        torch_model.network[2].bias.copy_(
            torch.from_numpy(
                numpy_model.layers[2].bias.data.copy()
            )
        )

    return (
        numpy_model,
        numpy_optimizer,
        torch_model,
        torch_optimizer
    )


# ==========================================
# ACCURACY FUNCTIONS
# ==========================================

def numpy_accuracy(model, X, y):

    correct = 0

    model.eval()

    for start in range(0, len(y), 256):

        end = min(start + 256, len(y))

        logits = model(
            Tensor(X[start:end])
        )

        predictions = np.argmax(
            logits.data,
            axis=1
        )

        correct += np.sum(
            predictions == y[start:end]
        )

    return correct / len(y)


def pytorch_accuracy(model, X, y):

    correct = 0

    model.eval()

    with torch.no_grad():

        for start in range(0, len(y), 256):

            end = min(start + 256, len(y))

            logits = model(X[start:end])

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == y[start:end]
            ).sum().item()

    return correct / len(y)


# ==========================================
# NUMPY TRAINING
# ==========================================

def train_numpy(
    model,
    optimizer,
    X_train,
    y_train,
    epoch_orders
):

    losses = []
    epoch_times = []

    model.train()

    training_start = time.perf_counter()

    for epoch in range(EPOCHS):

        epoch_start = time.perf_counter()

        indices = epoch_orders[epoch]

        total_loss = 0.0
        batches = 0

        for start in range(0, len(X_train), BATCH_SIZE):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            X_batch = Tensor(
                X_train[batch_indices]
            )

            y_batch = y_train[batch_indices]

            logits = model(X_batch)

            loss = cross_entropy(
                logits,
                y_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += float(loss.data)
            batches += 1

        average_loss = total_loss / batches

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        losses.append(average_loss)
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = (
        time.perf_counter() - training_start
    )

    return losses, epoch_times, total_training_time


# ==========================================
# PYTORCH TRAINING
# ==========================================

def train_pytorch(
    model,
    optimizer,
    X_train,
    y_train,
    epoch_orders
):

    criterion = nn.CrossEntropyLoss()

    losses = []
    epoch_times = []

    training_start = time.perf_counter()

    for epoch in range(EPOCHS):

        epoch_start = time.perf_counter()

        indices = epoch_orders[epoch]

        total_loss = 0.0
        batches = 0

        model.train()

        for start in range(0, len(X_train), BATCH_SIZE):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            batch_indices_t = torch.from_numpy(
                batch_indices
            )

            X_batch = X_train[batch_indices_t]
            y_batch = y_train[batch_indices_t]

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        average_loss = total_loss / batches

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        losses.append(average_loss)
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    total_training_time = (
        time.perf_counter() - training_start
    )

    return losses, epoch_times, total_training_time


# ==========================================
# NUMPY INFERENCE BENCHMARK
# ==========================================

def benchmark_numpy_inference(model, X):

    model.eval()

    # Convert input before timing.
    numpy_input = Tensor(X)

    # Warm-up without building computation graphs.
    with no_grad():
        for _ in range(10):
            model(numpy_input)

    timings = []

    # Benchmark inference without autograd overhead.
    with no_grad():

        for _ in range(INFERENCE_REPEATS):

            start = time.perf_counter()

            model(numpy_input)

            elapsed = time.perf_counter() - start

            timings.append(elapsed)

    return timings


# ==========================================
# PYTORCH INFERENCE BENCHMARK
# ==========================================

def benchmark_pytorch_inference(model, X):

    model.eval()

    # Warm-up

    with torch.no_grad():

        for _ in range(10):
            model(X)

        timings = []

        for _ in range(INFERENCE_REPEATS):

            start = time.perf_counter()

            model(X)

            elapsed = time.perf_counter() - start

            timings.append(elapsed)

    return timings


# ==========================================
# STATISTICS
# ==========================================

def calculate_statistics(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1))
        if len(values) > 1 else 0.0,
        "median": float(np.median(values))
    }


# ==========================================
# MAIN BENCHMARK
# ==========================================

def benchmark():

    print("\n" + "=" * 60)
    print("       NUMPY vs PYTORCH MULTI-TRIAL BENCHMARK")
    print("=" * 60)

    # --------------------------------------
    # Load MNIST
    # --------------------------------------

    X_train, y_train, X_test, y_test = load_mnist()

    X_train_t = torch.from_numpy(
        X_train
    ).to(DEVICE)

    y_train_t = torch.from_numpy(
        y_train
    ).to(DEVICE)

    X_test_t = torch.from_numpy(
        X_test
    ).to(DEVICE)

    y_test_t = torch.from_numpy(
        y_test
    ).to(DEVICE)

    print("\nDataset ready.")

    print(f"Trials: {TRIALS}")
    print(f"Epochs per trial: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    all_results = []

    # --------------------------------------
    # Run independent trials
    # --------------------------------------

    for trial in range(TRIALS):

        trial_seed = SEED + trial

        print("\n" + "=" * 60)
        print(f"TRIAL {trial + 1}/{TRIALS}")
        print("=" * 60)

        (
            numpy_model,
            numpy_optimizer,
            torch_model,
            torch_optimizer
        ) = create_models(trial_seed)

        # Generate identical shuffle orders

        rng = np.random.default_rng(trial_seed)

        epoch_orders = [
            rng.permutation(len(X_train))
            for _ in range(EPOCHS)
        ]

        print("\nIdentical initial weights configured.")
        print("Shared training shuffle orders configured.")

        # Alternate training order

        if trial % 2 == 0:

            print("\nTraining order: NumPy -> PyTorch")

            print("\nTRAINING NUMPY MODEL")

            (
                numpy_losses,
                numpy_epoch_times,
                numpy_training_time
            ) = train_numpy(
                numpy_model,
                numpy_optimizer,
                X_train,
                y_train,
                epoch_orders
            )

            print("\nTRAINING PYTORCH MODEL")

            (
                torch_losses,
                torch_epoch_times,
                torch_training_time
            ) = train_pytorch(
                torch_model,
                torch_optimizer,
                X_train_t,
                y_train_t,
                epoch_orders
            )

        else:

            print("\nTraining order: PyTorch -> NumPy")

            print("\nTRAINING PYTORCH MODEL")

            (
                torch_losses,
                torch_epoch_times,
                torch_training_time
            ) = train_pytorch(
                torch_model,
                torch_optimizer,
                X_train_t,
                y_train_t,
                epoch_orders
            )

            print("\nTRAINING NUMPY MODEL")

            (
                numpy_losses,
                numpy_epoch_times,
                numpy_training_time
            ) = train_numpy(
                numpy_model,
                numpy_optimizer,
                X_train,
                y_train,
                epoch_orders
            )

        # ----------------------------------
        # Final evaluation
        # ----------------------------------

        numpy_test_accuracy = numpy_accuracy(
            numpy_model,
            X_test,
            y_test
        )

        torch_test_accuracy = pytorch_accuracy(
            torch_model,
            X_test_t,
            y_test_t
        )

        # ----------------------------------
        # Inference benchmark
        # ----------------------------------

        X_sample = X_test[:INFERENCE_SAMPLES]

        X_sample_t = X_test_t[:INFERENCE_SAMPLES]

        numpy_inference_times = benchmark_numpy_inference(
            numpy_model,
            X_sample
        )

        torch_inference_times = benchmark_pytorch_inference(
            torch_model,
            X_sample_t
        )

        numpy_inference_time = float(
            np.median(numpy_inference_times)
        )

        torch_inference_time = float(
            np.median(torch_inference_times)
        )

        # ----------------------------------
        # Trial results
        # ----------------------------------

        print("\nNUMPY RESULTS")
        print("-" * 40)

        print(
            f"Training time: {numpy_training_time:.2f}s"
        )

        print(
            f"Test accuracy: {numpy_test_accuracy * 100:.2f}%"
        )

        print(
            f"Inference time: {numpy_inference_time:.6f}s"
        )

        print("\nPYTORCH RESULTS")
        print("-" * 40)

        print(
            f"Training time: {torch_training_time:.2f}s"
        )

        print(
            f"Test accuracy: {torch_test_accuracy * 100:.2f}%"
        )

        print(
            f"Inference time: {torch_inference_time:.6f}s"
        )

        all_results.append({

            "trial": trial + 1,

            "seed": trial_seed,

            "numpy": {
                "epoch_losses": numpy_losses,
                "epoch_times": numpy_epoch_times,
                "training_time": numpy_training_time,
                "test_accuracy": numpy_test_accuracy,
                "inference_time": numpy_inference_time
            },

            "pytorch": {
                "epoch_losses": torch_losses,
                "epoch_times": torch_epoch_times,
                "training_time": torch_training_time,
                "test_accuracy": torch_test_accuracy,
                "inference_time": torch_inference_time
            }

        })

    # ======================================
    # STATISTICAL ANALYSIS
    # ======================================

    numpy_training = [
        result["numpy"]["training_time"]
        for result in all_results
    ]

    torch_training = [
        result["pytorch"]["training_time"]
        for result in all_results
    ]

    numpy_accuracy_values = [
        result["numpy"]["test_accuracy"]
        for result in all_results
    ]

    torch_accuracy_values = [
        result["pytorch"]["test_accuracy"]
        for result in all_results
    ]

    numpy_inference = [
        result["numpy"]["inference_time"]
        for result in all_results
    ]

    torch_inference = [
        result["pytorch"]["inference_time"]
        for result in all_results
    ]

    summary = {

        "numpy": {
            "training_time": calculate_statistics(numpy_training),
            "test_accuracy": calculate_statistics(numpy_accuracy_values),
            "inference_time": calculate_statistics(numpy_inference)
        },

        "pytorch": {
            "training_time": calculate_statistics(torch_training),
            "test_accuracy": calculate_statistics(torch_accuracy_values),
            "inference_time": calculate_statistics(torch_inference)
        }

    }

    # ======================================
    # FINAL REPORT
    # ======================================

    print("\n" + "=" * 60)
    print("FINAL MULTI-TRIAL BENCHMARK REPORT")
    print("=" * 60)

    print(f"\nNumber of trials: {TRIALS}")

    # NumPy results

    print("\nNUMPY")
    print("-" * 40)

    print(
        "Training time: "
        f"{summary['numpy']['training_time']['mean']:.2f} "
        f"+/- {summary['numpy']['training_time']['std']:.2f}s"
    )

    print(
        "Test accuracy: "
        f"{summary['numpy']['test_accuracy']['mean'] * 100:.2f}% "
        f"+/- {summary['numpy']['test_accuracy']['std'] * 100:.2f}%"
    )

    print(
        "Inference time (median): "
        f"{summary['numpy']['inference_time']['mean']:.6f} "
        f"+/- {summary['numpy']['inference_time']['std']:.6f}s"
    )

    # PyTorch results

    print("\nPYTORCH")
    print("-" * 40)

    print(
        "Training time: "
        f"{summary['pytorch']['training_time']['mean']:.2f} "
        f"+/- {summary['pytorch']['training_time']['std']:.2f}s"
    )

    print(
        "Test accuracy: "
        f"{summary['pytorch']['test_accuracy']['mean'] * 100:.2f}% "
        f"+/- {summary['pytorch']['test_accuracy']['std'] * 100:.2f}%"
    )

    print(
        "Inference time (median): "
        f"{summary['pytorch']['inference_time']['mean']:.6f} "
        f"+/- {summary['pytorch']['inference_time']['std']:.6f}s"
    )

    # --------------------------------------
    # Speed comparison
    # --------------------------------------

    numpy_train_mean = (
        summary["numpy"]["training_time"]["mean"]
    )

    torch_train_mean = (
        summary["pytorch"]["training_time"]["mean"]
    )

    numpy_infer_mean = (
        summary["numpy"]["inference_time"]["mean"]
    )

    torch_infer_mean = (
        summary["pytorch"]["inference_time"]["mean"]
    )

    print("\nPERFORMANCE COMPARISON")
    print("-" * 40)

    print(
        f"PyTorch training speedup: "
        f"{numpy_train_mean / torch_train_mean:.2f}x"
    )

    print(
        f"PyTorch inference speedup: "
        f"{numpy_infer_mean / torch_infer_mean:.2f}x"
    )

    print("\n" + "=" * 60)

    # ======================================
    # SAVE RESULTS
    # ======================================

    results = {

        "configuration": {
            "trials": TRIALS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "architecture": "784 -> 128 -> 10",
            "device": "CPU",
            "cpu_threads": 1,
            "inference_samples": INFERENCE_SAMPLES,
            "inference_repeats": INFERENCE_REPEATS,
            "inference_statistic": "median",
            "seed": SEED
        },

        "individual_trials": all_results,

        "summary": summary

    }

    os.makedirs("artifacts", exist_ok=True)

    output_file = (
        "artifacts/numpy_vs_pytorch_results.json"
    )

    with open(output_file, "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("Results saved successfully!")

    print(f"File: {output_file}")


if __name__ == "__main__":
    benchmark()