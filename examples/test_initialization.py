import numpy as np

from src.nn.layers import Linear


def test_initialization(method):
    np.random.seed(42)

    layer = Linear(
        784,
        128,
        initialization=method
    )

    weights = layer.weight.data

    print(f"\nInitialization: {method}")
    print("-" * 40)
    print(f"Weight shape: {weights.shape}")
    print(f"Mean: {np.mean(weights):.6f}")
    print(f"Standard deviation: {np.std(weights):.6f}")
    print(f"Minimum: {np.min(weights):.6f}")
    print(f"Maximum: {np.max(weights):.6f}")


def main():
    print("WEIGHT INITIALIZATION EXPERIMENT")
    print("=" * 40)

    test_initialization("he")
    test_initialization("xavier")
    test_initialization("small")

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()