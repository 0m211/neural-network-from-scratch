import numpy as np
from contextlib import contextmanager


_GRAD_ENABLED = True


@contextmanager
def no_grad():
    global _GRAD_ENABLED

    previous_state = _GRAD_ENABLED
    _GRAD_ENABLED = False

    try:
        yield
    finally:
        _GRAD_ENABLED = previous_state


def unbroadcast(grad, target_shape):
    grad = np.asarray(grad, dtype=float)

    if target_shape == ():
        return np.asarray(grad.sum())

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)

        self._prev = set()
        self._backward = lambda: None

        self.requires_grad = _GRAD_ENABLED

    def _set_graph(self, parents, backward):
        if _GRAD_ENABLED:
            self._prev = {
                parent for parent in parents
                if parent.requires_grad
            }

            self._backward = backward

        return self

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data)

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        return out._set_graph((self, other), _backward)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data)

        def _backward():
            self.grad += unbroadcast(
                out.grad * other.data,
                self.data.shape
            )

            other.grad += unbroadcast(
                out.grad * self.data,
                other.data.shape
            )

        return out._set_graph((self, other), _backward)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Exponent must be an int or float.")

        out = Tensor(self.data ** exponent)

        def _backward():
            self.grad += (
                exponent
                * self.data ** (exponent - 1)
                * out.grad
            )

        return out._set_graph((self,), _backward)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** -1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(
                "This implementation supports 2D matrix multiplication only."
            )

        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                "Matrix dimensions are incompatible for multiplication."
            )

        out = Tensor(self.data @ other.data)

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        return out._set_graph((self, other), _backward)

    def relu(self):
        out = Tensor(np.maximum(0, self.data))

        def _backward():
            self.grad += (self.data > 0) * out.grad

        return out._set_graph((self,), _backward)

    def sigmoid(self):
        out = Tensor(
            np.exp(-np.logaddexp(0, -self.data))
        )

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        return out._set_graph((self,), _backward)

    def tanh(self):
        out = Tensor(np.tanh(self.data))

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        return out._set_graph((self,), _backward)

    def clip(self, min_value, max_value):
        if min_value > max_value:
            raise ValueError(
                "min_value must be less than or equal to max_value."
            )

        out = Tensor(
            np.clip(self.data, min_value, max_value)
        )

        def _backward():
            mask = (
                (self.data >= min_value)
                & (self.data <= max_value)
            )

            self.grad += mask * out.grad

        return out._set_graph((self,), _backward)

    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("Logarithm requires positive inputs.")

        out = Tensor(np.log(self.data))

        def _backward():
            self.grad += out.grad / self.data

        return out._set_graph((self,), _backward)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(
                axis=axis,
                keepdims=keepdims
            )
        )

        def _backward():
            grad = out.grad

            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)

                normalized_axes = tuple(
                    a if a >= 0 else a + self.data.ndim
                    for a in axes
                )

                for a in sorted(normalized_axes):
                    grad = np.expand_dims(grad, axis=a)

            self.grad += np.broadcast_to(
                grad,
                self.data.shape
            )

        return out._set_graph((self,), _backward)

    def mean(self, axis=None, keepdims=False):
        if self.data.size == 0:
            raise ValueError(
                "Cannot calculate the mean of an empty tensor."
            )

        if axis is None:
            count = self.data.size
        else:
            count = np.sum(
                np.ones_like(self.data),
                axis=axis,
                keepdims=keepdims
            )

        return self.sum(axis=axis, keepdims=keepdims) / count

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward() on a tensor created with no_grad()."
            )

        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output. Call sum() first."
            )

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        for node in topo:
            node.grad = np.zeros_like(node.data)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return (
            f"Tensor(data={self.data}, grad={self.grad})"
        )