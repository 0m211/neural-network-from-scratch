import numpy as np


class Dropout:

    def __init__(self, p=0.2, seed=None):

        if not 0.0 <= p < 1.0:
            raise ValueError(
                "Dropout probability must satisfy 0 <= p < 1."
            )

        self.p = p

        self.training = True

        self.rng = np.random.default_rng(seed)

        self.mask = None

    def train(self):

        self.training = True

        return self

    def eval(self):

        self.training = False

        self.mask = None

        return self

    def __call__(self, x):

        if not self.training or self.p == 0.0:

            self.mask = None

            return x

        keep_probability = 1.0 - self.p

        self.mask = (
            self.rng.random(x.data.shape) < keep_probability
        ).astype(x.data.dtype)

        self.mask = self.mask / keep_probability

        output = x * self.mask

        return output

    def parameters(self):

        return []