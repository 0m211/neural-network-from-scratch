import numpy as np

from src.autograd.tensor import Tensor
from src.nn.batch_norm import BatchNorm1d


def numerical_gradient(function, array, epsilon=1e-5):
    """
    Calculate numerical gradients using central differences.
    Restores each array element after perturbation.
    """
    gradients = np.zeros_like(array, dtype=np.float64)

    iterator = np.nditer(
        array,
        flags=["multi_index"],
        op_flags=["readwrite"]
    )

    while not iterator.finished:
        index = iterator.multi_index
        original_value = array[index]

        try:
            array[index] = original_value + epsilon
            positive_value = float(function())

            array[index] = original_value - epsilon
            negative_value = float(function())

            gradients[index] = (
                positive_value - negative_value
            ) / (2.0 * epsilon)

        finally:
            array[index] = original_value

        iterator.iternext()

    return gradients


def relative_error(analytical, numerical):
    numerator = np.linalg.norm(analytical - numerical)

    denominator = (
        np.linalg.norm(analytical)
        + np.linalg.norm(numerical)
        + 1e-12
    )

    return numerator / denominator


def main():
    np.random.seed(42)

    x_data = np.random.randn(4, 3).astype(np.float64)
    upstream = np.random.randn(4, 3).astype(np.float64)

    batch_norm = BatchNorm1d(3)
    batch_norm.train()

    print("BATCH NORMALIZATION TRAINING-MODE GRADIENT CHECK")
    print("-" * 55)

    # --------------------------------------------------
    # 1. Analytical gradients
    # --------------------------------------------------
    x = Tensor(x_data.copy())

    output = batch_norm(x)
    loss = (output * Tensor(upstream)).sum()

    loss.backward()

    analytical_input_grad = x.grad.copy()
    analytical_gamma_grad = batch_norm.gamma.grad.copy()
    analytical_beta_grad = batch_norm.beta.grad.copy()

    # --------------------------------------------------
    # 2. Numerical input gradient
    # --------------------------------------------------
    def input_loss():
        batch_norm.train()

        input_tensor = Tensor(x_data.copy())
        output_tensor = batch_norm(input_tensor)

        return np.sum(output_tensor.data * upstream)

    numerical_input_grad = numerical_gradient(
        input_loss,
        x_data
    )

    # --------------------------------------------------
    # 3. Numerical gamma gradient
    # --------------------------------------------------
    gamma_data = batch_norm.gamma.data.copy()

    def gamma_loss():
        batch_norm.train()

        input_tensor = Tensor(x_data.copy())
        output_tensor = batch_norm(input_tensor)

        return np.sum(output_tensor.data * upstream)

    numerical_gamma_grad = numerical_gradient(
        gamma_loss,
        batch_norm.gamma.data
    )

    batch_norm.gamma.data[...] = gamma_data

    # --------------------------------------------------
    # 4. Numerical beta gradient
    # --------------------------------------------------
    beta_data = batch_norm.beta.data.copy()

    def beta_loss():
        batch_norm.train()

        input_tensor = Tensor(x_data.copy())
        output_tensor = batch_norm(input_tensor)

        return np.sum(output_tensor.data * upstream)

    numerical_beta_grad = numerical_gradient(
        beta_loss,
        batch_norm.beta.data
    )

    batch_norm.beta.data[...] = beta_data

    # --------------------------------------------------
    # 5. Compare gradients
    # --------------------------------------------------
    checks = [
        (
            "Input",
            analytical_input_grad,
            numerical_input_grad
        ),
        (
            "Gamma",
            analytical_gamma_grad,
            numerical_gamma_grad
        ),
        (
            "Beta",
            analytical_beta_grad,
            numerical_beta_grad
        ),
    ]

    all_passed = True

    for name, analytical, numerical in checks:
        error = relative_error(analytical, numerical)

        print(f"\n{name} gradient:")
        print("Analytical:")
        print(analytical)

        print("Numerical:")
        print(numerical)

        print(f"Relative error: {error:.3e}")

        if error < 1e-5:
            print(f"PASS: {name} gradient is correct.")
        else:
            print(f"FAIL: {name} gradient mismatch.")
            all_passed = False

    print("\n" + "-" * 55)

    if all_passed:
        print("ALL TRAINING-MODE GRADIENT CHECKS PASSED!")
    else:
        print("SOME GRADIENT CHECKS FAILED.")
        raise AssertionError(
            "BatchNorm training-mode gradient check failed."
        )


if __name__ == "__main__":
    main()