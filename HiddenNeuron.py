from Neuron import Neuron


class HiddenNeuron(Neuron):
    x1 = 0
    x2 = 0
    w1 = 0
    w2 = 0

    def __init__(self, x1, x2, w1, w2):
        self.x1 = x1
        self.x2 = x2
        self.w1 = w1
        self.w2 = w2

    def calculate_value(self):
        """
        Calculates value of node
        :return: returns the value of the node
        """
        return self.x1 * self.w1 + self.x2 * self.w2

    def local_gradient(self, desired, actual):
        """
        Uses derivative of loss function to calculate local gradient
        :param desired: desired output
        :param actual: actual output
        :return: returns local gradient in order to check if weight needs to be increased or decreased
        """
        return -(desired - actual)
