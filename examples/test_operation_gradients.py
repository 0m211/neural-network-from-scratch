
import numpy as np

from src.autograd.tensor import Tensor
from src.nn.losses import mse_loss, binary_cross_entropy


def check_gradient(name, values, tensor_function, numpy_function):
    """
    Compare analytical gradients from autograd
    with numerical gradients.
    """

    x = Tensor(values.copy())

    # Analytical gradient using our autograd engine
    loss = tensor_function(x)
    loss.backward()
    analytical = x.grad.copy()

    # Numerical gradient using finite differences
    eps = 1e-5
    numerical = np.zeros_like(values, dtype=float)

    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()

        plus[index] += eps
        minus[index] -= eps

        loss_plus = numpy_function(plus)
        loss_minus = numpy_function(minus)

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * eps)

    difference = np.linalg.norm(analytical - numerical)

    print(f"\n{name}")
    print("Analytical:", analytical)
    print("Numerical: ", numerical)
    print("Difference:", difference)

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5
    ), f"{name} gradient check failed!"

    print("Passed!")


def test_relu():
    values = np.array([-2.0, -0.5, 1.0, 3.0])

    check_gradient(
        "ReLU",
        values,
        lambda x: x.relu().sum(),
        lambda a: np.maximum(a, 0).sum()
    )


def test_sigmoid():
    values = np.array([-2.0, 0.5, 2.0])

    check_gradient(
        "Sigmoid",
        values,
        lambda x: x.sigmoid().sum(),
        lambda a: (
            1 / (1 + np.exp(-a))
        ).sum()
    )


def test_mse():
    values = np.array([0.2, 0.6, 0.9])
    targets = np.array([0.0, 1.0, 1.0])

    check_gradient(
        "MSE",
        values,
        lambda x: mse_loss(x, Tensor(targets)),
        lambda a: np.mean((a - targets) ** 2)
    )


def test_binary_cross_entropy():
    values = np.array([0.2, 0.7, 0.9])
    targets = np.array([0.0, 1.0, 1.0])

    eps = 1e-7

    def numpy_bce(a):
        p = np.clip(a, eps, 1 - eps)

        return -np.mean(
            targets * np.log(p)
            + (1 - targets) * np.log(1 - p)
        )

    check_gradient(
        "Binary Cross-Entropy",
        values,
        lambda x: binary_cross_entropy(
            x, Tensor(targets)
        ),
        numpy_bce
    )


if __name__ == "__main__":
    test_relu()
    test_sigmoid()
    test_mse()
    test_binary_cross_entropy()

    print("\nAll operation gradient checks passed!")
