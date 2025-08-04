import os
import numpy as np
from layer import Layer
from utils import cross_entropy_loss, evaluate_accuracy, plot_image


class Network:
    def __init__(self, layer_sizes):
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers.")

        self.layers = []
        for i, size in enumerate(layer_sizes):
            use_softmax = (i == len(layer_sizes) - 1)
            self.layers.append(Layer(size, use_softmax=use_softmax))

        # Connect layers
        for i in range(len(self.layers) - 1):
            self.layers[i].connect_to(self.layers[i + 1])

    def forward(self, input_data):
        self.layers[0].load_data(input_data)
        self.layers[0].forward()
        return self.layers[-1].values

    def calculate_gradients(self, target):
        grads = {}
        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:  # Output layer
                grad_w, grad_b = layer.calculate_gradients(target)
            else:
                grad_w, grad_b = layer.calculate_gradients()
            grads[layer] = {"weights": grad_w, "biases": grad_b}
        return grads

    def train_batch(self, batch_images, batch_labels, learning_rate):
        grad_sums = {
            layer: {"weights": np.zeros_like(layer.weights), "biases": np.zeros_like(layer.biases)}
            for layer in self.layers[1:]
        }

        total_loss = 0.0

        for image, label in zip(batch_images, batch_labels):
            prediction = self.forward(image)
            target = np.eye(self.layers[-1].size)[label]
            total_loss += cross_entropy_loss(prediction, target)

            grads = self.calculate_gradients(target)
            for layer, grad in grads.items():
                grad_sums[layer]["weights"] += grad["weights"]
                grad_sums[layer]["biases"] += grad["biases"]

        batch_size = len(batch_images)
        for layer in self.layers[1:]:
            grad_sums[layer]["weights"] /= batch_size
            grad_sums[layer]["biases"] /= batch_size
            layer.weights -= learning_rate * grad_sums[layer]["weights"]
            layer.biases -= learning_rate * grad_sums[layer]["biases"]

        return total_loss / batch_size

    def test(self, images, labels, with_visuals=False):
        predicted_labels = []
        for image, label in zip(images, labels):
            prediction = self.forward(image)
            predicted = np.argmax(prediction)
            predicted_labels.append(predicted)

            print(f"Image with label {label} was {'correctly' if predicted == label else 'incorrectly'} classified as {predicted}.")
            if with_visuals:
                plot_image(image.reshape(28, 28), f"Correct: {label}, Predicted: {predicted}")

        accuracy = evaluate_accuracy(predicted_labels, labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_weights(self, directory="data"):
        os.makedirs(directory, exist_ok=True)
        for i, layer in enumerate(self.layers[1:], start=1):
            layer.save(f"{directory}/layer{i}")

    def load_weights(self, directory="data"):
        for i, layer in enumerate(self.layers[1:], start=1):
            path = f"{directory}/layer{i}.npz"
            if os.path.exists(path):
                layer.load(f"{directory}/layer{i}")
            else:
                print(f"Weight file not found: {path}")
                return False
        return True

