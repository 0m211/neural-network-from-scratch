from src.autograd.tensor import Tensor

print("TEST 1: Multiplication")

x = Tensor([2.0, 3.0])
loss = (x * x).mean()
loss.backward()

print("Loss:", loss.data)
print("Expected gradient: [2.0, 3.0]")
print("Actual gradient:", x.grad)


print("\nTEST 2: Sigmoid")

x = Tensor([0.0])
y = x.sigmoid()
y.backward()

print("Sigmoid output:", y.data)
print("Expected gradient: [0.25]")
print("Actual gradient:", x.grad)


print("\nTEST 3: Matrix multiplication")

x = Tensor([[1.0, 2.0]])
w = Tensor([[3.0], [4.0]])

loss = (x @ w).sum()
loss.backward()

print("Expected weight gradient: [[1.0], [2.0]]")
print("Actual weight gradient:", w.grad)


print("\nTEST 4: Logarithm")

x = Tensor([0.5])
loss = x.log().sum()
loss.backward()

print("Expected gradient: [2.0]")
print("Actual gradient:", x.grad)