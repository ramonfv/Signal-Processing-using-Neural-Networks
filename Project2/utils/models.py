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
        return np.where(x >= 0, 1, -1)
    
    def perceptron(self, x, weights):
        return self.activateFunction(np.dot(x, weights))
    
    def train_perceptron(self):
        for _ in range(self.epochs):
            for i in range(len(self.data)):
                x = self.data[i, :]
                y = self.perceptron(x, self.weights)
                error = self.labels[i] - y
                self.weights[:-1] += self.learning_rate * error * x[:-1]
                self.weights[-1] += self.learning_rate * error  # Atualização do bias
        return self.weights
    
    def predict(self, data, weights):
        data = np.hstack((np.ones((len(data), 1)), np.array(data)))
        return self.perceptron(data, weights)
    
    def evaluate(self, predictions, labels):
        correct_predictions = np.sum(predictions == labels)
        total_samples = len(labels)
        accuracy = correct_predictions / total_samples
        return accuracy

    def plot_3d(self, data, labels, predictions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, marker='o', label='Real - Class 1')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=predictions, marker='x', label='Predicted - Class 2')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
        Z = (-self.weights[0] - self.weights[1] * X - self.weights[2] * Y) / self.weights[3]
        ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.title('3D Plot of Predictions vs Real Labels')
        plt.legend()
        plt.show()

        
    