import numpy as np


class Node:
    def __init__(self, id):
        self.input_values = []
        self.itself_value = 0
        self.id = id
        self.sent = False
        self.n_of_connections = 0

    def set_value(self, value):
        self.itself_value = value

    def calculate_val(self):
        self.itself_value = sum(self.input_values)

    def sigmoid(self):
        self.itself_value = (2 / (1 + np.exp(-1 * self.itself_value))) - 1

    def logistic(self):
        self.itself_value = 1 / (1 + (np.e ** (-1 * self.itself_value)))
