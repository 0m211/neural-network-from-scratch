import numpy as np

from src.autograd.tensor import Tensor


EPSILON = 1e-5
RTOL = 1e-4
ATOL = 1e-6


def numerical_gradient(function, tensor, epsilon=EPSILON):
    """
    Calculate numerical gradients using central finite differences.

    Each element is perturbed slightly in the positive and negative
    directions. The resulting output difference estimates its gradient.
    """
    numerical_grad = np.zeros_like(tensor.data, dtype=np.float64)

    for index in np.ndindex(tensor.data.shape):
        original_value = tensor.data[index].item()

        try:
            tensor.data[index] = original_value + epsilon
            positive_output = float(function().data)

            tensor.data[index] = original_value - epsilon
            negative_output = float(function().data)

        finally:
            tensor.data[index] = original_value

        numerical_grad[index] = (
            positive_output - negative_output
        ) / (2.0 * epsilon)

    return numerical_grad


def check_gradient(name, function, inputs):
    """
    Compare analytical gradients from autograd with numerical gradients.
    """
    print(f"\nChecking: {name}")
    print("-" * 60)

    # Calculate analytical gradients.
    output = function()

    if output.data.size != 1:
        raise ValueError(
            f"{name}: function must return a scalar Tensor."
        )

    output.backward()

    analytical_gradients = [
        tensor.grad.copy() for tensor in inputs
    ]

    # Calculate numerical gradients independently.
    numerical_gradients = []

    for tensor in inputs:
        numerical_gradients.append(
            numerical_gradient(function, tensor)
        )

    # Compare results.
    all_passed = True

    for index, (analytical, numerical) in enumerate(
        zip(analytical_gradients, numerical_gradients),
        start=1
    ):
        print(f"Input tensor {index}:")
        print("Analytical gradient:")
        print(analytical)

        print("Numerical gradient:")
        print(numerical)

        difference = np.max(
            np.abs(analytical - numerical)
        )

        print(f"Maximum absolute difference: {difference:.10e}")

        if not np.allclose(
            analytical,
            numerical,
            rtol=RTOL,
            atol=ATOL
        ):
            all_passed = False
            print("FAIL: Gradients do not match.")
        else:
            print("PASS: Gradients match.")

    if not all_passed:
        raise AssertionError(
            f"Gradient check failed for: {name}"
        )

    print(f"SUCCESS: {name}")


def test_matrix_multiplication():
    """Check gradients for matrix multiplication."""
    x = Tensor([
        [0.2, -0.4],
        [0.7, 0.3]
    ])

    weights = Tensor([
        [0.5, -0.2],
        [0.1, 0.8]
    ])

    bias = Tensor([0.1, -0.1])

    def function():
        predictions = x @ weights + bias
        return (predictions * predictions).mean()

    check_gradient(
        "Matrix multiplication with bias and MSE",
        function,
        [x, weights, bias]
    )


def test_nonlinear_computation_graph():
    """Check gradients through a graph containing nonlinear operations."""
    x = Tensor([
        [0.2, -0.3],
        [0.5, 0.7]
    ])

    weights = Tensor([
        [0.4, -0.2],
        [0.3, 0.6]
    ])

    bias = Tensor([0.1, 0.2])

    def function():
        z = x @ weights + bias
        activated = z.tanh()
        return (activated * activated).mean()

    check_gradient(
        "Matrix multiplication, bias, tanh, and MSE",
        function,
        [x, weights, bias]
    )


def test_shared_computation_graph():
    """Check gradient accumulation when a tensor is reused."""
    x = Tensor([0.4, -0.6, 0.8])

    def function():
        first_path = x * x
        second_path = x * 3.0
        return (first_path + second_path).sum()

    check_gradient(
        "Shared computation graph",
        function,
        [x]
    )


def main():
    print("=" * 65)
    print("ADVANCED NUMERICAL GRADIENT CHECKS")
    print("=" * 65)

    test_matrix_multiplication()
    test_nonlinear_computation_graph()
    test_shared_computation_graph()

    print("\n" + "=" * 65)
    print("ALL ADVANCED GRADIENT CHECKS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()