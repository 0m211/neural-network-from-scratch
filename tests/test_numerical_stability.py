import numpy as np

from src.autograd.tensor import Tensor
from src.nn.dropout import Dropout
from src.optim.optimizers import SGD, RMSProp, Adam


def test_sigmoid_extreme_values():
    """Sigmoid should remain finite for extreme inputs."""

    positive = Tensor([1000.0])
    negative = Tensor([-1000.0])

    positive_output = positive.sigmoid()
    negative_output = negative.sigmoid()

    assert np.all(np.isfinite(positive_output.data))
    assert np.all(np.isfinite(negative_output.data))

    assert np.allclose(positive_output.data, [1.0])
    assert np.allclose(negative_output.data, [0.0])

    print("PASS: Sigmoid handles extreme values")


def test_tanh_extreme_values():
    """Tanh should remain finite for extreme inputs."""

    positive = Tensor([1000.0])
    negative = Tensor([-1000.0])

    positive_output = positive.tanh()
    negative_output = negative.tanh()

    assert np.all(np.isfinite(positive_output.data))
    assert np.all(np.isfinite(negative_output.data))

    assert np.allclose(positive_output.data, [1.0])
    assert np.allclose(negative_output.data, [-1.0])

    print("PASS: Tanh handles extreme values")


def test_optimizer_rejects_invalid_learning_rates():
    """Optimizers should reject invalid learning rates."""

    optimizers = [
        lambda: SGD([], lr=-0.01),
        lambda: RMSProp([], lr=0.0),
        lambda: Adam([], lr=-1.0),
    ]

    for create_optimizer in optimizers:
        try:
            create_optimizer()
        except (ValueError, TypeError):
            continue

        raise AssertionError(
            "An optimizer accepted an invalid learning rate."
        )

    print("PASS: Optimizers reject invalid learning rates")


def test_dropout_rejects_invalid_probability():
    """Dropout probability must be in the valid range."""

    invalid_probabilities = [-0.1, 1.0, 1.5]

    for probability in invalid_probabilities:
        try:
            Dropout(p=probability)
        except (ValueError, TypeError):
            continue

        raise AssertionError(
            f"Dropout accepted invalid probability: {probability}"
        )

    print("PASS: Dropout rejects invalid probabilities")


def test_nan_and_infinity_detection():
    """Verify that tests can identify non-finite tensor values."""

    invalid_values = Tensor([
        np.nan,
        np.inf,
        -np.inf
    ])

    assert not np.all(np.isfinite(invalid_values.data))

    print("PASS: Non-finite values can be detected")


def main():
    print("=" * 65)
    print("NUMERICAL STABILITY AND INPUT VALIDATION TESTS")
    print("=" * 65)

    tests = [
        test_sigmoid_extreme_values,
        test_tanh_extreme_values,
        test_optimizer_rejects_invalid_learning_rates,
        test_dropout_rejects_invalid_probability,
        test_nan_and_infinity_detection,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as error:
            failed += 1
            print(f"FAIL: {test.__name__}: {error}")

    print("\n" + "=" * 65)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print("=" * 65)

    if failed:
        raise SystemExit(1)

    print("ALL NUMERICAL STABILITY TESTS PASSED!")


if __name__ == "__main__":
    main()