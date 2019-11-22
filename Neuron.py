import numpy as np


class NeuralNetwork:
    x1 = 0
    x2 = 0
    w1 = 0
    w2 = 0

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def hidden_layer(self):
        """
        Calculates value of node
        :return: returns the value of the node
        """
        return self.x1 * self.w1 + self.x2 * self.w2

    def output_layer(self, x):
        """
        Passes value of hidden layers to sigmoid activation function to "map" input to certain range of outputs
        :param x: output of hidden layers
        :return: returns mapped output
        """
        return 1 / (1 + np.exp(-x))

    def loss(self, desired, actual):
        """
        Calculates how  wrong output of neural network is, uses squared-error loss function
        :param desired: desired output
        :param actual: actual output
        :return: returns how far the actual output is from the desired output
        """
        return 0.5 * np.power((desired - actual), 2)

    def backpropagation(self, desired, actual):
        """
        Uses derivative of loss function to calculate local gradient
        :param desired: desired output
        :param actual: actual output
        :return: returns local gradient in order to check if weight needs to be increased or decreased
        """
        return -(desired - actual)


if __name__ == '__main__':
    network = NeuralNetwork(0, 0)
    hidden = network.hidden_layer()
    result = network.output_layer(hidden)
    loss = network.loss(0, result)
    print(result)
    print(loss)
