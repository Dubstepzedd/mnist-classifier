import numpy as np

class Neuron:
    def __init__(self):
        self.value = 0.0
        self.incoming_connections = []
        self.outgoing_connections = []
        self.inputs = []

    def receive_input(self, input_value: float):
        self.inputs.append(input_value)

    def calculate_output(self):
        # Sum all inputs and apply activation function
        if self.inputs:
            total_input = sum(self.inputs)
            self.value = self.activation_function(total_input)
            self.inputs = []  # Clear for next forward pass

    def add_outgoing_connection(self, connection):
        self.outgoing_connections.append(connection)

    def add_incoming_connection(self, connection):
        self.incoming_connections.append(connection)

    def activation_function(self, x):
        # Simple sigmoid activation
        return 1 / (1 + np.exp(-x))

    def feed_forward(self):
        for connection in self.outgoing_connections:
            connection.transmit_signal()

    def __repr__(self):
        return f"Neuron(value={self.value}, input_sum={sum(self.inputs)})"
