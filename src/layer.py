from typing import Optional
import numpy as np
from utils import sigmoid, derivative_cross_entropy_softmax, derivative_sigmoid, softmax

class Layer:
    def __init__(self, size: int, use_softmax: bool = False):
        if size <= 0:
            raise ValueError("Layer size must be a positive integer.")

        # Actual layer properties
        self.size = size
        self.values = np.zeros(size)
        self.biases = np.random.uniform(-0.5, 0.5, size)
        self.weights = None
        self.next_layer = None
        self.prev_layer = None
        self.delta = None  # Used for backpropagation
        self.z = None
        self.use_softmax = use_softmax

    def connect_to(self, next_layer: 'Layer'):
        self.next_layer = next_layer
        next_layer.prev_layer = self
        next_layer.set_weights(np.random.uniform(-0.5, 0.5, (next_layer.size, self.size)))

    def save(self, name: str):
        np.savez(f"{name}.npz", weights=self.weights, biases=self.biases)

    def load(self, name: str):
        data = np.load(f"{name}.npz")
        self.weights = data["weights"]
        self.biases = data["biases"]

    def load_data(self, data: np.array):
        if len(data.shape) != 1:
            flattened_data = data.flatten()
        else:
            flattened_data = data
        if len(flattened_data) != self.size:
            raise ValueError(f"Data length ({len(flattened_data)}) must match layer size ({self.size})")

        self.values = flattened_data.copy()

    def calculate_gradients(self, wanted_values: np.ndarray = None):
        if self.prev_layer is None:
            raise ValueError("Cannot calculate gradients without a previous layer.")

        if wanted_values is not None:  # output layer
            delta = derivative_cross_entropy_softmax(self.z, wanted_values)
        else:
            delta_next = self.next_layer.delta
            weights_next = self.next_layer.weights
            da_dz = derivative_sigmoid(self.z)
            delta = (weights_next.T @ delta_next) * da_dz

        self.delta = delta

        # So this combines two vectors into a matrix, if delta has shape (n,) and prev_layer.values has shape (m,), then grad_weights will have shape (n, m) - didn't learn this operation in Linear Algebra sadly :/
        grad_weights = np.outer(delta, self.prev_layer.values)
        grad_bias = delta

        return grad_weights, grad_bias


    def forward(self, input_values: Optional[np.ndarray] = None):
        if input_values is not None and self.weights is not None:
            linear_output = self.weights @ input_values + self.biases
            self.z = linear_output
            if self.use_softmax:
                self.values = softmax(linear_output) # Apparently this gives better results and is only to be used on the output layer
            else:
                self.values = sigmoid(linear_output)

        if self.next_layer is not None:
            self.next_layer.forward(self.values)

    def set_weights(self, weights: np.ndarray):
        if weights.ndim != 2:
            raise ValueError("Weights must be a 2D array")
        if weights.shape[0] != self.size:
            raise ValueError(f"Weight matrix first dimension ({weights.shape[0]}) must match layer size ({self.size})")
        self.weights = weights.copy()

    def print_state(self):
        print(f"Layer values: {self.values}")
        if self.weights is not None:
            print(f"Weights shape: {self.weights.shape}")
            print(f"Bias: {self.biases}")

    def __repr__(self):
        weight_info = f", weights_shape={self.weights.shape}" if self.weights is not None else ""
        return f"Layer(size={self.size}{weight_info})"

