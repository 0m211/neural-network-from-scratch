import numpy as np

from src.utils.data_split import train_validation_test_split


def main():
    print("=" * 65)
    print("TRAIN / VALIDATION / TEST SPLIT TESTS")
    print("=" * 65)

    rng = np.random.default_rng(123)

    X = rng.normal(size=(1000, 5))
    y = np.array([0] * 700 + [1] * 300)

    X_train, X_val, X_test, y_train, y_val, y_test = (
        train_validation_test_split(
            X,
            y,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
            stratify=True,
        )
    )

    # Check split sizes.
    assert len(X_train) + len(X_val) + len(X_test) == 1000

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    # Check that all samples are assigned exactly once.
    original_rows = {
        tuple(row) for row in X
    }

    split_rows = (
        [tuple(row) for row in X_train]
        + [tuple(row) for row in X_val]
        + [tuple(row) for row in X_test]
    )

    assert len(split_rows) == 1000
    assert len(set(split_rows)) == 1000
    assert set(split_rows) == original_rows

    print("PASS: No samples were lost or duplicated.")

    # Check approximate class proportions.
    for name, labels in [
        ("Training", y_train),
        ("Validation", y_val),
        ("Test", y_test),
    ]:
        positive_ratio = np.mean(labels == 1)

        print(
            f"{name} positive-class ratio: "
            f"{positive_ratio:.3f}"
        )

        assert abs(positive_ratio - 0.30) < 0.05

    print("PASS: Class proportions are approximately preserved.")

    # Check reproducibility.
    split_a = train_validation_test_split(
        X, y, random_state=42
    )

    split_b = train_validation_test_split(
        X, y, random_state=42
    )

    for array_a, array_b in zip(split_a, split_b):
        assert np.array_equal(array_a, array_b)

    print("PASS: Identical seeds produce identical splits.")

    # Check invalid ratios.
    try:
        train_validation_test_split(
            X,
            y,
            train_ratio=0.8,
            validation_ratio=0.3,
            test_ratio=0.1,
        )
    except ValueError:
        print("PASS: Invalid split ratios are rejected.")
    else:
        raise AssertionError(
            "Invalid split ratios were accepted."
        )

    print("\nALL DATA-SPLIT TESTS PASSED!")


if __name__ == "__main__":
    main()