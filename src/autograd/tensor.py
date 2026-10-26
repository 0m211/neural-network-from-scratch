import numpy as np


def unbroadcast(grad, target_shape):
    """
    Reduce a broadcast gradient to the original shape
    of its corresponding input.
    """
    grad = np.asarray(grad, dtype=float)

    # Handle scalar inputs.
    if target_shape == ():
        return np.asarray(grad.sum())

    # Remove extra leading dimensions.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Reduce dimensions that were broadcast from size 1.
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)

        # Parent tensors in the computation graph.
        self._prev = set()

        # Local backward function.
        self._backward = lambda: None

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Negation
    def __neg__(self):
        return self * -1

    # Subtraction
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other - self

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(
                out.grad * other.data,
                self.data.shape
            )

            other.grad += unbroadcast(
                out.grad * self.data,
                other.data.shape
            )

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Power
    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                "Exponent must be an int or float."
            )

        out = Tensor(self.data ** exponent)
        out._prev = {self}

        def _backward():
            self.grad += (
                exponent
                * self.data ** (exponent - 1)
                * out.grad
            )

        out._backward = _backward
        return out

    # Division
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * (self ** -1)

    # Matrix multiplication
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(
                "This implementation supports 2D matrix "
                "multiplication only."
            )

        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                "Matrix dimensions are incompatible for multiplication."
            )

        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    # ReLU activation
    def relu(self):
        out = Tensor(np.maximum(0, self.data))
        out._prev = {self}

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    # Sigmoid activation
    def sigmoid(self):
        out = Tensor(
            np.exp(-np.logaddexp(0, -self.data))
        )
        out._prev = {self}

        def _backward():
            self.grad += (
                out.data * (1 - out.data) * out.grad
            )

        out._backward = _backward
        return out

    # Tanh activation
    def tanh(self):
        out = Tensor(np.tanh(self.data))
        out._prev = {self}

        def _backward():
            self.grad += (
                (1 - out.data ** 2) * out.grad
            )

        out._backward = _backward
        return out

    # Clip values while preserving the computation graph
    def clip(self, min_value, max_value):
        if min_value > max_value:
            raise ValueError(
                "min_value must be less than or equal to max_value."
            )

        out = Tensor(
            np.clip(self.data, min_value, max_value)
        )
        out._prev = {self}

        def _backward():
            mask = (
                (self.data >= min_value)
                & (self.data <= max_value)
            )

            self.grad += mask * out.grad

        out._backward = _backward
        return out

    # Natural logarithm
    def log(self):
        if np.any(self.data <= 0):
            raise ValueError(
                "Logarithm requires positive inputs."
            )

        out = Tensor(np.log(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    # Sum tensor elements
    def sum(self, axis=None, keepdims=False):
        """
        Sum tensor elements.

        Parameters:
            axis: Axis or axes along which to sum.
                  None means sum all elements.
            keepdims: If True, retain reduced dimensions.
        """
        out = Tensor(
            self.data.sum(
                axis=axis,
                keepdims=keepdims
            )
        )

        out._prev = {self}

        def _backward():
            grad = out.grad

            # Restore dimensions removed during summation.
            if axis is not None and not keepdims:
                if isinstance(axis, tuple):
                    axes = axis
                else:
                    axes = (axis,)

                # Normalize negative axes.
                normalized_axes = tuple(
                    a if a >= 0 else a + self.data.ndim
                    for a in axes
                )

                # Expand dimensions in ascending order.
                for a in sorted(normalized_axes):
                    grad = np.expand_dims(grad, axis=a)

            # Broadcast the gradient to the original shape.
            self.grad += np.broadcast_to(
                grad,
                self.data.shape
            )

        out._backward = _backward
        return out

    # Mean of tensor elements
    def mean(self, axis=None, keepdims=False):
        """
        Calculate the mean of tensor elements.

        Parameters:
            axis: Axis or axes along which to calculate the mean.
                  None means calculate the mean of all elements.
            keepdims: If True, retain reduced dimensions.
        """
        if self.data.size == 0:
            raise ValueError(
                "Cannot calculate the mean of an empty tensor."
            )

        if axis is None:
            count = self.data.size
        else:
            # Number of elements contributing to each mean.
            count = np.sum(
                np.ones_like(self.data),
                axis=axis,
                keepdims=keepdims
            )

        return (
            self.sum(axis=axis, keepdims=keepdims)
            / count
        )

    # Backpropagation
    def backward(self):
        """
        Calculate gradients through the computation graph.

        The output must contain exactly one element.
        Call sum() before backward() if the output has
        multiple elements.
        """
        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output. "
                "Call sum() first."
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

        # Reset gradients for this backward pass.
        for node in topo:
            node.grad = np.zeros_like(node.data)

        # Gradient of the output with respect to itself.
        self.grad = np.ones_like(self.data)

        # Propagate gradients backward.
        for node in reversed(topo):
            node._backward()

    # Display tensor values and gradients
    def __repr__(self):
        return (
            f"Tensor(data={self.data}, grad={self.grad})"
        )