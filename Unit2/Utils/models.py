
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, data, labels, learning_rate, epochs=None):
        if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("Os dados e os rótulos devem ser arrays NumPy")
        self.data = np.hstack((np.ones((len(data), 1)), data))
        self.labels = labels
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(data.shape[1] + 1)
    
    def activateFunction(self, x):
        return np.where(x >= 0, 1, 0)
    
    def perceptron(self, x, weights):
        return self.activateFunction(np.dot(x, weights))
    
    def train_perceptron(self):
        for _ in range(self.epochs):
            for i in range(len(self.data)):
                x = self.data[i, :]
                y = self.perceptron(x, self.weights)
                error = self.labels[i] - y
                self.weights += self.learning_rate * error * x
        return self.weights
    
    def predict(self, data, weights):
        data = np.hstack((np.ones((len(data), 1)), np.array(data)))
        return self.perceptron(data, weights)
    
    def plot(self):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(len(self.data)):
                x1, x2 = self.data[i][1], self.data[i][2]
                ax.scatter(x1, x2, self.labels[i], color = 'r' if self.labels[i] == 1 else 'b')

            x1Range = np.linspace(self.data[:, 1].min(), self.data[:, 1].max(), num=20)
            x2Range = np.linspace(self.data[:, 2].min(), self.data[:, 2].max(), num=20)
            X1, X2 = np.meshgrid(x1Range, x2Range)
            Zponit = (-self.weights[0] - self.weights[1]*X1 - self.weights[2]*X2) / self.weights[-1]

            ax.plot_surface(X1, X2, Zponit, alpha=0.5, rstride=100, cstride=100, color='y', edgecolor='none')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            plt.show()