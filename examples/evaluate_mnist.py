import numpy as np

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.model_io import load_model
from examples.load_mnist import load_mnist


def create_model():
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )
    return model


def predict_batch(model, images):
    inputs = Tensor(images)
    logits = model(inputs).data
    predictions = np.argmax(logits, axis=1)
    return predictions


def evaluate_model(model, X_test, y_test, batch_size=256):
    total_correct = 0
    total_images = len(X_test)

    all_predictions = []

    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)

        images = X_test[start:end]
        actual_labels = y_test[start:end]

        predictions = predict_batch(model, images)

        correct = np.sum(predictions == actual_labels)
        total_correct += correct

        all_predictions.extend(predictions.tolist())

        print(
            f"Evaluated {end}/{total_images} images | "
            f"Batch accuracy: "
            f"{correct / len(images) * 100:.2f}%"
        )

    all_predictions = np.array(all_predictions)

    accuracy = total_correct / total_images

    return accuracy, all_predictions


def main():
    print("\nLoading MNIST dataset...")

    _, _, X_test, y_test = load_mnist()

    print("\nCreating neural network...")

    model = create_model()

    print("\nLoading trained model...")

    load_model(model, "mnist_model.npz")

    print("\nEvaluating all test images...")

    accuracy, predictions = evaluate_model(
        model,
        X_test,
        y_test,
        batch_size=256
    )

    print("\n" + "=" * 45)
    print("         MNIST EVALUATION RESULTS")
    print("=" * 45)

    print(f"Total test images: {len(y_test)}")
    print(f"Correct predictions: {np.sum(predictions == y_test)}")
    print(f"Incorrect predictions: {np.sum(predictions != y_test)}")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")

    print("\nFirst 20 predictions:")
    print("-" * 45)
    print(f"{'Image':<10}{'Actual':<12}{'Predicted':<12}{'Result'}")
    print("-" * 45)

    for i in range(20):
        actual = int(y_test[i])
        predicted = int(predictions[i])

        if actual == predicted:
            result = "Correct"
        else:
            result = "Wrong"

        print(
            f"{i:<10}{actual:<12}{predicted:<12}{result}"
        )

    incorrect_indices = np.where(predictions != y_test)[0]

    print("\nFirst 10 incorrect predictions:")
    print("-" * 45)

    if len(incorrect_indices) == 0:
        print("No incorrect predictions found!")
    else:
        for index in incorrect_indices[:10]:
            print(
                f"Image {index}: "
                f"Actual = {y_test[index]}, "
                f"Predicted = {predictions[index]}"
            )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()