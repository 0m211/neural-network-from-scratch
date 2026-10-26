import numpy as np

from src.autograd.tensor import Tensor


def numerical_gradient(function, values, eps=1e-5):
    """
    Estimate each gradient using central finite differences.
    """

    values = np.asarray(values, dtype=float)
    gradients = np.zeros_like(values)

    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()

        plus[index] += eps
        minus[index] -= eps

        loss_plus = function(plus)
        loss_minus = function(minus)

        gradients[index] = (
            loss_plus - loss_minus
        ) / (2 * eps)

    return gradients


def test_gradient():
    values = np.array([1.5, -2.0, 3.0])

    # Analytical gradient from our Tensor engine.
    x = Tensor(values.copy())

    loss = (x * x + 2 * x).sum()
    loss.backward()

    analytical = x.grad.copy()

    # Independent numerical approximation.
    def calculate_loss(array):
        return np.sum(array * array + 2 * array)

    numerical = numerical_gradient(
        calculate_loss,
        values
    )

    print("Analytical gradients:", analytical)
    print("Numerical gradients: ", numerical)

    difference = np.linalg.norm(
        analytical - numerical
    )

    print("Gradient difference:", difference)

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5
    ), "Gradient check failed!"

    print("Gradient check passed!")


if __name__ == "__main__":
    test_gradient()