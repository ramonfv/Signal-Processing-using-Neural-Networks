from models import Perceptron
import numpy as np

data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
labels = np.array([0, 0, 0, 1])

learning_rate = 0.5
number_weights = 4

perceptron = Perceptron(data, labels, epochs=10, learning_rate=learning_rate, number_weights=number_weights)

perceptron.run()