from typing import Optional
import numpy as np

class Layer:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("Layer size must be a positive integer.")

        self.size = size
        self.values = np.zeros(size)
        self.biases = np.random.uniform(-0.5, 0.5, size)
        self.next_layer = None

    def connect_to(self, next_layer: 'Layer'):
        self.next_layer = next_layer
        next_layer.set_weights(np.random.uniform(-0.5, 0.5, (next_layer.size, self.size)))

    def load_data(self, data: np.array):
        flattened_data = data.flatten()
        if len(flattened_data) != self.size:
            raise ValueError(f"Data length ({len(flattened_data)}) must match layer size ({self.size})")

        self.values = flattened_data.copy()

    def forward(self, input_values: Optional[np.array] = None):
        if input_values is not None and self.weights is not None:
            # Apply weights and biases, then activation
            linear_output = self.weights @ input_values + self.biases
            self.values = self._activation_function(linear_output)

        # Trigger next layer's forward pass if connected
        if self.next_layer is not None:
            self.next_layer.forward(self.values)

    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

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
            print(f"Biases: {self.biases}")

    def __repr__(self):
        weight_info = f", weights_shape={self.weights.shape}" if self.weights is not None else ""
        return f"Layer(size={self.size}{weight_info})"

