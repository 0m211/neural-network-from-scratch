import numpy as np
import matplotlib.pyplot as plt

from src.autograd.tensor import Tensor
from src.nn.layers import Linear
from src.nn.activations import ReLU
from src.nn.sequential import Sequential
from src.nn.model_io import load_model
from examples.load_mnist import load_mnist


def create_model():
    return Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )


def predict_digit(model, image):
    image_batch = image.reshape(1, 784)

    logits = model(Tensor(image_batch)).data

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_logits)

    probabilities = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    predicted_digit = int(np.argmax(probabilities[0]))
    confidence = float(probabilities[0, predicted_digit])

    return predicted_digit, confidence, probabilities[0]


def main():
    print("Loading MNIST dataset...")

    _, _, X_test, y_test = load_mnist()

    print("MNIST dataset loaded successfully.")

    print("Creating neural network...")

    model = create_model()

    print("Loading trained model...")

    load_model(model, "mnist_model.npz")

    print("Model loaded successfully.")

    image_index = 0

    image = X_test[image_index]
    actual_digit = int(y_test[image_index])

    print("Predicting digit...")

    predicted_digit, confidence, probabilities = predict_digit(
        model, image
    )

    print("\n========== PREDICTION RESULT ==========")
    print(f"Image index:       {image_index}")
    print(f"Actual digit:      {actual_digit}")
    print(f"Predicted digit:   {predicted_digit}")
    print(f"Model confidence:  {confidence * 100:.2f}%")

    print("\nScores for each digit:")

    for digit, probability in enumerate(probabilities):
        print(f"Digit {digit}: {probability * 100:.2f}%")

    if predicted_digit == actual_digit:
        print("\nSUCCESS: The prediction is correct!")
    else:
        print("\nThe prediction is incorrect.")

    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(
        f"Actual: {actual_digit} | Predicted: {predicted_digit}"
    )
    plt.axis("off")

    plt.savefig("mnist_prediction.png", bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nPrediction image saved to: mnist_prediction.png")


if __name__ == "__main__":
    main()