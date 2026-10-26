import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.model_io import save_model, load_model

print("TEST_SAVE_LOAD FILE STARTED")

def create_model():
    """Create a neural network for testing model saving and loading."""


    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )


def main():
    np.random.seed(42)


# 1. Create a model and generate test predictions.
    print("Creating original model...")

    model = create_model()

    X = np.random.rand(4, 784).astype(np.float64)
    original_predictions = model(Tensor(X)).data.copy()

    print("Original model created successfully.")

    # 2. Save the model parameters.
    print("\nSaving model parameters...")

    save_model(model, "save_load_test.npz")

# 3. Create a new model with the same architecture.
    print("\nCreating a new model...")

    new_model = create_model()

# 4. Load the saved parameters.
    print("\nLoading saved parameters...")

    load_model(new_model, "save_load_test.npz")

# 5. Generate predictions using the restored model.
    print("\nGenerating predictions using restored model...")

    restored_predictions = new_model(Tensor(X)).data

# 6. Verify that both predictions match.
    print("\nComparing predictions...")

    np.testing.assert_allclose(
    original_predictions,
    restored_predictions,
    rtol=1e-10,
    atol=1e-10
)

    print("\nSUCCESS: Save/load test passed!")
    print("Original and restored predictions match.")
    print("Model saving and loading work correctly!")


if __name__ == "__main__":
    print("MAIN FUNCTION STARTED")
    main()
