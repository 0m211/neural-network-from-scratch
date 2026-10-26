import numpy as np
from src.autograd.tensor import Tensor


class Linear:

    def __init__(
        self,
        in_features,
        out_features,
        initialization="he"
    ):
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "in_features and out_features must be positive."
            )

        if initialization == "he":
            scale = np.sqrt(2.0 / in_features)

        elif initialization == "xavier":
            scale = np.sqrt(
                2.0 / (in_features + out_features)
            )

        elif initialization == "small":
            scale = 0.01

        else:
            raise ValueError(
                "Unknown initialization method. "
                "Choose 'he', 'xavier', or 'small'."
            )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Tensor(
            np.random.randn(
                in_features,
                out_features
            ) * scale
        )

        self.bias = Tensor(
            np.zeros(out_features)
        )

    def __call__(self, x):

        # Convert NumPy arrays into custom Tensor objects.
        if not isinstance(x, Tensor):
            x = Tensor(x)

        # Validate input dimensions.
        if x.data.ndim != 2:
            raise ValueError(
                f"Linear layer expects a 2D input, "
                f"but received shape {x.data.shape}."
            )

        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Input feature mismatch: expected "
                f"{self.in_features} features, but received "
                f"{x.data.shape[1]}."
            )

        # Validate weight dimensions.
        expected_shape = (
            self.in_features,
            self.out_features
        )

        if self.weight.data.shape != expected_shape:
            raise ValueError(
                f"Invalid weight shape: expected "
                f"{expected_shape}, but received "
                f"{self.weight.data.shape}."
            )

        # Forward pass.
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]