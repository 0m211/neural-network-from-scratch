
import numpy as np

from src.autograd.tensor import Tensor


def mse_loss(predictions, targets):
    """
    Calculate Mean Squared Error (MSE).

    MSE = mean((predictions - targets) ** 2)
    """

    if predictions.data.shape != targets.data.shape:
        raise ValueError(
            "Predictions and targets must have the same shape."
        )

    difference = predictions - targets
    return (difference ** 2).mean()


def binary_cross_entropy(predictions, targets, eps=1e-7):
    """
    Calculate Binary Cross-Entropy (BCE).

    Predictions must be probabilities between 0 and 1.
    Targets must contain only 0 and 1.
    """

    # Validate shapes.
    if predictions.data.shape != targets.data.shape:
        raise ValueError(
            "Predictions and targets must have the same shape."
        )

    # Validate epsilon.
    if not 0 < eps < 0.5:
        raise ValueError(
            "eps must be between 0 and 0.5."
        )

    # Validate predictions.
    if not np.all(np.isfinite(predictions.data)):
        raise ValueError(
            "Predictions must contain finite values."
        )

    if not np.all(
        (predictions.data >= 0)
        & (predictions.data <= 1)
    ):
        raise ValueError(
            "Predictions must be between 0 and 1."
        )

    # Validate targets.
    if not np.all(np.isfinite(targets.data)):
        raise ValueError(
            "Targets must contain finite values."
        )

    # FIX: Check every target independently.
    valid_targets = np.isin(targets.data, [0.0, 1.0])

    if not np.all(valid_targets):
        raise ValueError(
            f"Targets must contain only 0 and 1. "
            f"Received: {np.unique(targets.data)}"
        )

    # Clip probabilities without disconnecting the graph.
    probabilities = predictions.clip(eps, 1.0 - eps)

    # Calculate BCE.
    positive_loss = targets * probabilities.log()

    negative_loss = (
        (1.0 - targets)
        * (1.0 - probabilities).log()
    )

    loss = -(positive_loss + negative_loss)

    return loss.mean()


import numpy as np
from src.autograd.tensor import Tensor


def cross_entropy(logits, targets):
    """
    Multiclass cross-entropy loss.

    Parameters
    ----------
    logits : Tensor
        Raw model outputs with shape (batch_size, num_classes).

    targets : NumPy array
        Integer class labels with shape (batch_size,).
        Example: [7, 2, 1, 0]

    Returns
    -------
    Tensor
        Scalar mean cross-entropy loss.
    """

    if logits.data.ndim != 2:
        raise ValueError(
            "Logits must have shape (batch_size, num_classes)."
        )
    if not np.all(np.isfinite(logits.data)):
        raise ValueError(
            "Logits must contain only finite values."
        )
    
    targets = np.asarray(targets)

    batch_size, num_classes = logits.data.shape

    if batch_size == 0:
        raise ValueError(
            "Cross-entropy requires a non-empty batch."
        )

    if num_classes == 0:
        raise ValueError(
            "Cross-entropy requires at least one class."
        )

    if targets.shape != (batch_size,):
        raise ValueError(
            "Targets must have shape (batch_size,)."
        )

    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Targets must contain integer class labels.")

    if np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("Target class index is out of range.")

    # Numerically stable log-sum-exp calculation.
    # The maximum is treated as a constant for this operation.
    shifted = (
        logits.data
        - np.max(logits.data, axis=1, keepdims=True)
    )

    exp_shifted = np.exp(shifted)
    probabilities = (
        exp_shifted
        / np.sum(exp_shifted, axis=1, keepdims=True)
    )

    log_sum_exp = np.log(
        np.sum(exp_shifted, axis=1)
    ) + np.max(logits.data, axis=1)

    correct_class_logits = logits.data[
        np.arange(batch_size), targets
    ]

    loss_value = np.mean(
        log_sum_exp - correct_class_logits
    )

    result = Tensor(np.array(loss_value))

    result._prev = {logits}

    def _backward():
        grad_logits = probabilities.copy()

        grad_logits[
            np.arange(batch_size), targets
        ] -= 1.0

        grad_logits /= batch_size

        logits.grad += result.grad * grad_logits

    result._backward = _backward

    return result

