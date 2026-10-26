from src.autograd.value import Value

x = Value(0.0)

y = x.sigmoid()

y.backward()

print("sigmoid(0) =", y.data)
print("gradient =", x.grad)