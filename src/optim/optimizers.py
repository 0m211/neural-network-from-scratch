import numpy as np


def _validate_parameters(parameters):
    """Validate and store model parameters."""
    parameters = list(parameters)

    for parameter in parameters:
        if not hasattr(parameter, "data") or not hasattr(parameter, "grad"):
            raise TypeError(
                "Each parameter must have both data and grad attributes."
            )

        if parameter.data.shape != parameter.grad.shape:
            raise ValueError(
                "Parameter data and gradient shapes must match."
            )

    return parameters


def _validate_positive(name, value):
    """Require a finite, strictly positive numeric value."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number.")


def _validate_non_negative(name, value):
    """Require a finite, non-negative numeric value."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a finite non-negative number."
        )


def _validate_unit_interval(name, value):
    """Require a finite value in [0, 1)."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError(f"{name} must be in the range [0, 1).")


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters, lr=0.01):
        _validate_positive("lr", lr)

        self.parameters = _validate_parameters(parameters)
        self.lr = float(lr)

    def zero_grad(self):
        """Reset gradients to zero."""
        for parameter in self.parameters:
            parameter.grad.fill(0)

    def step(self):
        """Update parameters using their gradients."""
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


class RMSProp:
    """RMSProp optimizer."""

    def __init__(
        self,
        parameters,
        lr=0.001,
        rho=0.9,
        eps=1e-8
    ):
        _validate_positive("lr", lr)
        _validate_unit_interval("rho", rho)
        _validate_positive("eps", eps)

        self.parameters = _validate_parameters(parameters)
        self.lr = float(lr)
        self.rho = float(rho)
        self.eps = float(eps)

        self.cache = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

    def zero_grad(self):
        """Reset gradients to zero."""
        for parameter in self.parameters:
            parameter.grad.fill(0)

    def step(self):
        """Update parameters using RMSProp."""
        for parameter, cache in zip(self.parameters, self.cache):
            cache *= self.rho
            cache += (1 - self.rho) * (parameter.grad ** 2)

            parameter.data -= (
                self.lr
                * parameter.grad
                / (np.sqrt(cache) + self.eps)
            )


class Adam:
    """
    Adam optimizer with bias correction and optional L2-style
    weight decay coupled to the gradient.
    """

    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0
    ):
        _validate_positive("lr", lr)
        _validate_unit_interval("beta1", beta1)
        _validate_unit_interval("beta2", beta2)
        _validate_positive("eps", eps)
        _validate_non_negative("weight_decay", weight_decay)

        self.parameters = _validate_parameters(parameters)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        self.t = 0

        self.m = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

        self.v = [
            np.zeros_like(parameter.data)
            for parameter in self.parameters
        ]

    def zero_grad(self):
        """Reset gradients to zero."""
        for parameter in self.parameters:
            parameter.grad.fill(0)

    def step(self):
        """Update parameters using Adam and optional weight decay."""
        self.t += 1

        for parameter, m, v in zip(
            self.parameters,
            self.m,
            self.v
        ):
            # Copy the gradient so the parameter gradient is not modified.
            gradient = parameter.grad.copy()

            # Add coupled L2-style weight decay.
            if self.weight_decay > 0:
                gradient += self.weight_decay * parameter.data

            # Update exponential moving averages.
            m *= self.beta1
            m += (1 - self.beta1) * gradient

            v *= self.beta2
            v += (1 - self.beta2) * (gradient ** 2)

            # Correct initialization bias.
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters.
            parameter.data -= (
                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.eps)
            )