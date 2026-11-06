import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    print("Downloading/loading MNIST dataset...")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff"
    )

    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    X /= 255.0

    X_train = X[:60000]
    y_train = y[:60000]

    X_test = X[60000:]
    y_test = y[60000:]

    print("\nDataset loaded successfully!")
    print("Training data:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing data:", X_test.shape)
    print("Testing labels:", y_test.shape)

    print("\nPixel range:")
    print("Minimum:", X_train.min())
    print("Maximum:", X_train.max())

    print("\nFirst 10 labels:")
    print(y_train[:10])

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_mnist()