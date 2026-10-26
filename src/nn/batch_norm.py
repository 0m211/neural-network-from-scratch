import numpy as np

from src.autograd.tensor import Tensor


class BatchNorm1d:

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        if num_features <= 0:
            raise ValueError("num_features must be positive.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between 0 and 1.")

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = Tensor(np.ones((1, num_features), dtype=np.float64))
        self.beta = Tensor(np.zeros((1, num_features), dtype=np.float64))

        self.running_mean = np.zeros((1, num_features), dtype=np.float64)
        self.running_var = np.ones((1, num_features), dtype=np.float64)

    def __call__(self, x):
        if len(x.data.shape) != 2:
            raise ValueError("BatchNorm1d expects a 2D input.")

        if x.data.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, "
                f"received {x.data.shape[1]}."
            )

        if self.training:
            batch_size = x.data.shape[0]

            if batch_size == 0:
                raise ValueError("BatchNorm1d cannot process an empty batch.")

            batch_mean = x.mean(axis=0, keepdims=True)
            centered = x - batch_mean

            batch_var = (centered ** 2).mean(
                axis=0,
                keepdims=True
            )

            normalized = centered / ((batch_var + self.eps) ** 0.5)

            self.running_mean = (
                (1.0 - self.momentum) * self.running_mean
                + self.momentum * batch_mean.data
            )

            if batch_size > 1:
                running_var_batch = batch_var.data * (
                    batch_size / (batch_size - 1)
                )
            else:
                running_var_batch = batch_var.data

            self.running_var = (
                (1.0 - self.momentum) * self.running_var
                + self.momentum * running_var_batch
            )

        else:
            normalized = (
                x - self.running_mean
            ) / ((self.running_var + self.eps) ** 0.5)

        return self.gamma * normalized + self.beta

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return [self.gamma, self.beta]