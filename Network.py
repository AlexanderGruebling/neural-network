import numpy as np
from HiddenNeuron import HiddenNeuron
from OutputNeuron import OutputNeuron


class NeuralNetwork:
    x1 = 0
    x2 = 0
    w1 = 0
    w2 = 0
    DESIRED = 0

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        hidden_neuron = HiddenNeuron(self.x1, self.x2, self.w1, self.w2)
        hidden = hidden_neuron.calculate_value()
        output_neuron = OutputNeuron(hidden)
        sigmoid = output_neuron.calculate_value()
        print("Input Data: x1={0}, x2={1}".format(self.x1, self.x2))
        print("Desired Output: {0}".format(self.DESIRED))
        print("Output: {0}".format(output_neuron.calculate_value()))
        print("Loss: {0}".format(self.loss(self.DESIRED, sigmoid)))
        print("Local Gradient: {0}".format(hidden_neuron.local_gradient(self.DESIRED, sigmoid)))

    def loss(self, desired, actual):
        """
        Calculates how  wrong output of neural network is, uses squared-error loss function
        :param desired: desired output
        :param actual: actual output
        :return: returns how far the actual output is from the desired output
        """
        return 0.5 * np.power((desired - actual), 2)

    def backpropagation(self, local, upstream):
        return upstream * local

    def local_gradient(self, desired, actual):
        """
        Uses derivative of loss function to calculate local gradient
        :param desired: desired output
        :param actual: actual output
        :return: returns local gradient in order to check if weight needs to be increased or decreased
        """
        return -(desired - actual)


if __name__ == '__main__':
    network = NeuralNetwork(0, 0)
