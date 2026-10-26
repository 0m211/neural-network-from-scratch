import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.model_io import load_model

from examples.load_mnist import load_mnist


MODEL_PATH = "artifacts/learning_curves_best.npz"
BATCH_SIZE = 256
NUM_IMAGES = 25


def get_predictions(model, X, batch_size=256):
    model.eval()

    predictions = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))

        X_batch = Tensor(X[start:end])
        logits = model(X_batch)

        batch_predictions = np.argmax(
            logits.data,
            axis=1
        )

        predictions.extend(batch_predictions.tolist())

    return np.array(predictions, dtype=np.int64)


def main():
    print("=" * 65)
    print("MISCLASSIFIED MNIST IMAGE ANALYSIS")
    print("=" * 65)

    print("\nLoading MNIST dataset...")

    _, _, X_test, y_test = load_mnist()

    print("\nRebuilding model...")

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )

    print("\nLoading best checkpoint...")

    load_model(model, MODEL_PATH)

    print("\nGenerating predictions...")

    y_pred = get_predictions(
        model,
        X_test,
        BATCH_SIZE
    )

    incorrect_indices = np.where(y_pred != y_test)[0]

    print(f"\nTotal test images: {len(y_test)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}")

    if len(incorrect_indices) == 0:
        print("No incorrect predictions found.")
        return

    selected_indices = incorrect_indices[:NUM_IMAGES]

    num_images = len(selected_indices)
    cols = 5
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 2.5 * rows)
    )

    axes = np.array(axes).reshape(-1)

    print("\nSelected misclassified images:")

    for plot_index, image_index in enumerate(selected_indices):
        image = X_test[image_index].reshape(28, 28)

        actual = int(y_test[image_index])
        predicted = int(y_pred[image_index])

        axes[plot_index].imshow(
            image,
            cmap="gray"
        )

        axes[plot_index].set_title(
            f"Actual: {actual}\nPredicted: {predicted}"
        )

        axes[plot_index].axis("off")

        print(
            f"Test index: {image_index:5d} | "
            f"Actual: {actual} | "
            f"Predicted: {predicted}"
        )

    for index in range(num_images, len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        "Misclassified MNIST Images",
        fontsize=16
    )

    plt.tight_layout()

    output_path = "artifacts/misclassified_images.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nImage visualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()