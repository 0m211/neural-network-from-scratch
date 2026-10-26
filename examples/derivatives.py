'''def multiplication_derivative(x, y):
    """
    z = x * y

    dz/dx = y
    dz/dy = x
    """

    dz_dx = y
    dz_dy = x

    return dz_dx, dz_dy


def addition_derivative(x, y):
    """
    z = x + y

    dz/dx = 1
    dz/dy = 1
    """
    dz_dx = 1
    dz_dy = 1

    return dz_dx, dz_dy


# Example
x = 2
y = 3

dz_dx, dz_dy = multiplication_derivative(x, y)

print("For z = x * y")
print("dz/dx =", dz_dx)
print("dz/dy =", dz_dy)


dz_dx, dz_dy = addition_derivative(x, y)

print("\nFor z = x + y")
print("dz/dx =", dz_dx)
print("dz/dy =", dz_dy)

'''
x = Value(2)
y = Value(3)

z = x * y

z.backward()

print(x.grad)  # 3
print(y.grad)  # 2
