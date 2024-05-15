
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
            

            
class MLP(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer, 
                 learning_rate, epochs, max_error):
        super(MLP, self).__init__()
        
        self.hidden = nn.Linear(input_layer, hidden_layer)
        self.output = nn.Linear(hidden_layer, output_layer)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_error = max_error
        self.initializeWeights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.accuracy  = []
        self.loss = []
        
    def initializeWeights(self):
        nn.init.normal_(self.hidden.weight)
        nn.init.normal_(self.output.weight)
        nn.init.normal_(self.hidden.bias)
        nn.init.normal_(self.output.bias)
        
    def updateWeights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name or 'bias' in name:
                    param -= self.learning_rate * param.grad
        
    def activationFunction(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def lossFunction(self, y, y_pred):
        return torch.mean((y - y_pred)**2)
    
    def forward(self, x):
        x = self.activationFunction(self.hidden(x))
        x = self.activationFunction(self.output(x))
        return x
    
    def backward(self, x, y):
        for _ in range(self.epochs):
            y_pred = self.forward(x)
            loss = self.lossFunction(y, y_pred)
            if loss <= self.max_error:
                break
            self.optimizer.zero_grad()
            loss.backward()
            self.updateWeights()
            
        return self.parameters()
    
    def train(self, data, labels):
        data = data.unsqueeze(1)
        for epoch in range(self.epochs):
            out = self.forward(data)
            loss = self.lossFunction(labels, out)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch} - Loss: {loss.item()}')
                
                
    def predict(self, data):
        data = data.unsqueeze(1)
        data = torch.transpose(data, 0,1)
        with torch.no_grad():
            prediction = self.forward(data)
        return prediction
                
    def evaluate(self, data, labels):
        out = self.forward(data)
        loss =  self.lossFunction(labels, out)
        with torch.no_grad():
            predictions = self.predict(data)
            binary_predictions = predictions.round()
            correct = (binary_predictions == labels).sum().item()
            total = labels.numel()
            self.accuracy = correct / total
            
        print(f'Test Loss: {loss.item()}\n Accuracy: {self.accuracy}')