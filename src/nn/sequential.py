class Sequential:
    """
    Sequential neural network model.

    Passes input through each layer in order and collects
    trainable parameters from all layers.
    """

    def __init__(self, *layers):
        self.layers = layers
        self.training = True

    def __call__(self, x):
        """
        Forward pass through all layers.
        """
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        """
        Return all trainable parameters from every layer.
        """
        params = []

        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())

        return params

    def zero_grad(self):
        """
        Reset gradients for all trainable parameters.
        """
        for param in self.parameters():
            param.grad.fill(0.0)

    def train(self):
        """
        Switch the model and its layers to training mode.
        """
        self.training = True

        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

        return self

    def eval(self):
        """
        Switch the model and its layers to evaluation mode.
        """
        self.training = False

        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

        return self