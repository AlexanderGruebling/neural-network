import numpy as np
from Neuron import Neuron


class OutputNeuron(Neuron):
    x = 0

    def __init__(self, x):
        self.x = x

    def calculate_value(self):
        """
        Passes value of hidden layers to sigmoid activation function to "map" input to certain range of outputs
        :param x: output of hidden layers
        :return: returns mapped output
        """
        return 1 / (1 + np.exp(-self.x))
