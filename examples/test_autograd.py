from src.autograd.value import Value


x = Value(2)
y = Value(3)

z = x * y

z.backward()

print("z =", z.data)
print("dz/dx =", x.grad)
print("dz/dy =", y.grad)