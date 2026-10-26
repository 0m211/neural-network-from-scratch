from src.autograd.value import Value


def numerical_gradient(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)



x = Value(2.0)
y = Value(3.0)

a = x * y
f = a ** 2

f.backward()

print("f =", f.data)
print("df/dx =", x.grad)
print("df/dy =", y.grad)


