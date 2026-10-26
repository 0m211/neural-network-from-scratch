import random

from src.autograd.value import Value


class Neuron:

    def __init__(self, n_inputs, activation="tanh"):

        # Each connection has a trainable weight.
        self.weights = [
            Value(random.uniform(-1, 1))
            for _ in range(n_inputs)
        ]

        self.bias = Value(0.0)
        self.activation = activation

    def __call__(self, inputs):

        # Calculate weighted sum + bias.
        output = self.bias

        for weight, value in zip(self.weights, inputs):
            output = output + weight * value

        # Apply activation function.
        if self.activation == "tanh":
            return output.tanh()

        if self.activation == "sigmoid":
            return output.sigmoid()

        if self.activation == "relu":
            return output.relu()

        return output

    def parameters(self):
        return self.weights + [self.bias]




class Layer:

    def __init__(self, n_inputs, n_outputs, activation):

        self.neurons = [
            Neuron(n_inputs, activation)
            for _ in range(n_outputs)
        ]

    def __call__(self, inputs):

        return [
            neuron(inputs)
            for neuron in self.neurons
        ]

    def parameters(self):

        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]    



class MLP:

    def __init__(self, sizes):

        self.layers = []

        for i in range(len(sizes) - 1):

            # Use sigmoid in the final layer.
            if i == len(sizes) - 2:
                activation = "sigmoid"
            else:
                activation = "tanh"

            self.layers.append(
                Layer(sizes[i], sizes[i + 1], activation)
            )

    def __call__(self, inputs):

        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def parameters(self):

        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]    



# Make initialization reproducible.
random.seed(7)

# XOR training dataset.
X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

targets = [0.0, 1.0, 1.0, 0.0]

# Create a network: 2 inputs -> 4 hidden neurons -> 1 output.
model = MLP([2, 4, 1])

learning_rate = 0.5
epochs = 3000

for epoch in range(epochs + 1):

    # Forward pass for all training examples.
    losses = []

    for inputs, target in zip(X, targets):

        prediction = model(inputs)[0]

        # Squared error for this example.
        loss = (prediction - target) ** 2

        losses.append(loss)

    # Mean squared error across the four examples.
    total_loss = sum(losses, Value(0.0)) / len(losses)

    # Backpropagation: calculate gradients.
    total_loss.backward()

    # Gradient descent: update parameters.
    for parameter in model.parameters():

        parameter.data -= learning_rate * parameter.grad

    # Display training progress.
    if epoch % 500 == 0:

        print(
            f"Epoch: {epoch:4d} | "
            f"Loss: {total_loss.data:.6f}"
        )


# Evaluate the trained network.
print("\nFinal predictions:")

for inputs, target in zip(X, targets):

    prediction = model(inputs)[0].data

    print(
        f"Input: {inputs} | "
        f"Prediction: {prediction:.4f} | "
        f"Target: {target:.0f}"
    )    