
import math


class Value:

    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set()
        self._backward = lambda: None

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # Power
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), (
            "Only int and float exponents are supported"
        )

        out = Value(self.data ** exponent)
        out._prev = {self}

        def _backward():
            self.grad += (
                exponent
                * self.data ** (exponent - 1)
                * out.grad
            )

        out._backward = _backward
        return out

    # Negation
    def __neg__(self):
        return self * -1

    # Subtraction
    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    # Division
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    # Reverse operations
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1)

    # Exponential
    def exp(self):
        out = Value(math.exp(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # Sigmoid activation
    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    # Backpropagation
    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        # Reset gradients for this backward pass
        for node in topo:
            node.grad = 0.0

        # The output's gradient with respect to itself is 1
        self.grad = 1.0

        # Propagate gradients backward
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t)
        out._prev = {self}

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
