import random
from neuron import Neuron

class Connection:
    def __init__(self, source: Neuron, target: Neuron):
        self.source = source
        self.target = target
        self.weight = random.uniform(-0.5, 0.5)

    def set_weight(self, weight: float):
        self.weight = weight

    def get_weight(self) -> float:
        return self.weight

    def transmit_signal(self):
        signal = self.source.value
        input_value = signal * self.weight
        self.target.receive_input(input_value)
