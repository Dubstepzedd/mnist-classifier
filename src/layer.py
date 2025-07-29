from typing import List

import numpy as np
from connection import Connection
from neuron import Neuron


class Layer:
    def __init__(self, size: int = 0):
        self.nodes: List[Neuron] = []
        if size > 0:
            self.create_neurons(size)

    def add_node(self, node: Neuron):
        self.nodes.append(node)

    def connect_to(self, other_layer: 'Layer'):
        for node in self.nodes:
            for other_node in other_layer.nodes:
                conn = Connection(node, other_node)
                node.add_outgoing_connection(conn)
                other_node.add_incoming_connection(conn)

    def create_neurons(self, count: int):
        for _ in range(count):
            self.add_node(Neuron())

    def feed_data(self, data: np.array):
        flattened_data = data.flatten()
        if len(flattened_data) != len(self.nodes):
            raise ValueError("Data length must match the number of nodes in the layer.")

        for node, value in zip(self.nodes, flattened_data):
            node.value = value

    def feed_forward(self):
        for node in self.nodes:
            node.calculate_output()

        for node in self.nodes:
            node.feed_forward()

    def print_state(self):
        for i, node in enumerate(self.nodes):
            print(f"Node {i}: {node}")

    def __repr__(self):
        return f"Layer(size={len(self.nodes)})"

