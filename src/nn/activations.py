from src.autograd.tensor import Tensor


class ReLU:
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []


class Sigmoid:
    def __call__(self, x):
        return x.sigmoid()

    def parameters(self):
        return []