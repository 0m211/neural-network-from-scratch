
import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    print("Downloading/loading MNIST dataset...")

    # Fetch the original MNIST handwritten digit dataset.
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False
    )

    # Images: (70000, 784)
    X = mnist.data.astype(np.float32)

    # Labels: (70000,)
    y = mnist.target.astype(np.int64)

    # Normalize pixel values from [0, 255] to [0, 1].
    X /= 255.0

    # Original dataset order:
    # First 60000 samples are training data.
    # Last 10000 samples are test data.
    X_train = X[:60000]
    y_train = y[:60000]

    X_test = X[60000:]
    y_test = y[60000:]

    print("\nDataset loaded successfully!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    print("\nPixel range:")
    print("Minimum:", X_train.min())
    print("Maximum:", X_train.max())

    print("\nFirst 10 labels:")
    print(y_train[:10])

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_mnist()

