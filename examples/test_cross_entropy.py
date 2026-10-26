
import numpy as np

from src.autograd.tensor import Tensor
from src.nn.losses import cross_entropy


def numerical_gradient(function, values, eps=1e-5):
    gradients = np.zeros_like(values, dtype=float)

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


def numpy_cross_entropy(logits, targets):
    # Stable calculation independent of the Tensor implementation.
    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )

    exp_values = np.exp(shifted)

    log_sum_exp = (
        np.log(exp_values.sum(axis=1))
        + np.max(logits, axis=1)
    )

    correct_logits = logits[
        np.arange(len(targets)), targets
    ]

    return np.mean(log_sum_exp - correct_logits)


def test_cross_entropy():
    logits_data = np.array([
        [3.0, 1.0, 0.2],
        [0.1, 2.5, 0.3],
        [0.2, 0.1, 2.0]
    ])

    targets = np.array([0, 1, 2])

    logits = Tensor(logits_data.copy())

    loss = cross_entropy(logits, targets)
    loss.backward()

    analytical = logits.grad.copy()

    numerical = numerical_gradient(
        lambda values: numpy_cross_entropy(
            values, targets
        ),
        logits_data
    )

    difference = np.linalg.norm(
        analytical - numerical
    )

    print("Loss:", loss.data)
    print("\nAnalytical gradients:\n", analytical)
    print("\nNumerical gradients:\n", numerical)
    print("\nGradient difference:", difference)

    assert np.allclose(
        analytical,
        numerical,
        atol=1e-5,
        rtol=1e-5
    ), "Cross-entropy gradient check failed!"

    print("\nCross-entropy gradient check passed!")


if __name__ == "__main__":
    test_cross_entropy()

