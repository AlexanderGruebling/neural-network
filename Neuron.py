from abc import ABC, abstractmethod


class Neuron(ABC):

    @abstractmethod
    def calculate_value(self):
        return
