import numpy as np

from src.autograd.tensor import Tensor
from src.optim.optimizers import SGD, RMSProp, Adam


def test_optimizer(optimizer_class, lr):
    # Create a simple parameter.
    weight = Tensor([5.0])

    # We want weight to approach 0.
    optimizer = optimizer_class([weight], lr=lr)

    initial_value = weight.data.copy()

    for _ in range(100):
        optimizer.zero_grad()

        # Loss = weight squared.
        loss = (weight * weight).mean()

        # Calculate gradient.
        loss.backward()

        # Update weight.
        optimizer.step()

    final_value = weight.data[0]

    print(f"\nOptimizer: {optimizer_class.__name__}")
    print(f"Initial weight: {initial_value[0]:.6f}")
    print(f"Final weight:   {final_value:.6f}")

    assert abs(final_value) < abs(initial_value[0]), (
        f"{optimizer_class.__name__} failed to reduce the loss."
    )

    print("Test passed!")


test_optimizer(SGD, lr=0.01)
test_optimizer(RMSProp, lr=0.01)
test_optimizer(Adam, lr=0.1)

print("\nAll optimizer tests passed!")