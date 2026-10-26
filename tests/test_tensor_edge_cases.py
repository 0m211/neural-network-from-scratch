import numpy as np

from src.autograd.tensor import Tensor


def assert_close(actual, expected, message=""):
    """Check whether two arrays are numerically close."""
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-7):
        raise AssertionError(
            f"{message}\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual}"
        )


def test_broadcasting_gradients():
    """Test gradients when adding a vector to a matrix."""
    x = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    bias = Tensor([0.5, 1.0, 1.5])

    loss = (x + bias).sum()
    loss.backward()

    assert_close(
        x.grad,
        np.ones((2, 3)),
        "Incorrect gradient for matrix input."
    )

    assert_close(
        bias.grad,
        np.array([2.0, 2.0, 2.0]),
        "Incorrect gradient for broadcast bias."
    )

    print("PASS: Broadcasting gradients")


def test_matrix_multiplication_gradients():
    """Test gradients for two-dimensional matrix multiplication."""
    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    weights = Tensor([
        [0.5, 1.0],
        [1.5, 2.0]
    ])

    loss = (x @ weights).sum()
    loss.backward()

    expected_x_grad = np.array([
        [1.5, 3.5],
        [1.5, 3.5]
    ])

    expected_weights_grad = np.array([
        [4.0, 4.0],
        [6.0, 6.0]
    ])

    assert_close(
        x.grad,
        expected_x_grad,
        "Incorrect matrix multiplication input gradient."
    )

    assert_close(
        weights.grad,
        expected_weights_grad,
        "Incorrect matrix multiplication weight gradient."
    )

    print("PASS: Matrix multiplication gradients")


def test_shared_computation_graph():
    """Test gradients when one tensor is used by multiple paths."""
    x = Tensor([2.0, 3.0])

    squared = x * x
    doubled = x * 2.0

    loss = (squared + doubled).sum()
    loss.backward()

    # For each element:
    # d(x*x + 2*x)/dx = 2*x + 2
    expected_gradient = np.array([6.0, 8.0])

    assert_close(
        x.grad,
        expected_gradient,
        "Incorrect gradient for shared computation graph."
    )

    print("PASS: Shared computation graph")


def test_scalar_broadcasting():
    """Test gradients when adding a scalar to a matrix."""
    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0]
    ])

    scalar = Tensor(2.0)

    loss = (x + scalar).sum()
    loss.backward()

    assert_close(
        x.grad,
        np.ones((2, 2)),
        "Incorrect gradient for matrix."
    )

    assert_close(
        scalar.grad,
        np.array(4.0),
        "Incorrect gradient for scalar."
    )

    print("PASS: Scalar broadcasting")


def main():
    print("=" * 60)
    print("TENSOR EDGE-CASE TESTS")
    print("=" * 60)

    test_broadcasting_gradients()
    test_matrix_multiplication_gradients()
    test_shared_computation_graph()
    test_scalar_broadcasting()

    print("\nAll Tensor edge-case tests passed!")


if __name__ == "__main__":
    main()