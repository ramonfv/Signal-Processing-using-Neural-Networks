import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt    

class LSTMModel(nn.Module):
    def __init__(self, inputFeatures, hiddenFeatures, layerDimension, outputDimension):
        super(LSTMModel, self).__init__()
        self.hiddenFeatures = hiddenFeatures
        self.layerDimension = layerDimension   
        self.lstm = nn.LSTM(inputFeatures, hiddenFeatures, layerDimension, batch_first=True)
        self.fc = nn.Linear(hiddenFeatures, outputDimension)

    # def forward(self, x):
    #     # h0 = torch.zeros(self.layerDimension, x.size(0), self.hiddenFeatures).requires_grad_()
    #     # c0 = torch.zeros(self.layerDimension, x.size(0), self.hiddenFeatures).requires_grad_()
    #     # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
    #     x, (h, c) = self.lstm(x)
    #     x = self.fc(x)
        
    #     return x
        

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :]  # Select only the last output of the sequence
        x = self.fc(x)
        return x
class GRUModel(nn.Module):
    def __init__(self, inputFeatures, hiddenFeatures, layerDimension, outputDimension):
        super(GRUModel, self).__init__()
        self.hiddenFeatures = hiddenFeatures
        self.layerDimension = layerDimension   
        self.gru = nn.GRU(inputFeatures, hiddenFeatures, layerDimension, batch_first=True)
        self.fc = nn.Linear(hiddenFeatures, outputDimension)

    def forward(self, x):
        h0 = torch.zeros(self.layerDimension, x.size(0), self.hiddenFeatures).requires_grad_()
        out, (hn) = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out

def prepareSequences(data, seqLength):
    xs, ys = [], []
    for i in range(len(data) - seqLength):
        x = data[i:i+seqLength]
        y = data[i+seqLength]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def trainModel(model, xTrain, yTrain, xVal, yVal, numEpochs, learningRate, batchSize):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    
    trainDataset = torch.utils.data.TensorDataset(xTrain, yTrain)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    
    valDataset = torch.utils.data.TensorDataset(xVal, yVal)
    valLoader = torch.utils.data.DataLoader(dataset=valDataset, batch_size=batchSize, shuffle=False)
    
    for epoch in range(numEpochs):
        model.train()
        for i, (inputs, targets) in enumerate(trainLoader):
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        valLoss = 0.0
        with torch.no_grad():
            for inputs, targets in valLoader:
                valOutputs = model(inputs)
                valLoss += criterion(valOutputs, targets).item()
        
        valLoss /= len(valLoader)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}, Val Loss: {valLoss:.4f}')
    return model

def evaluateModel(model, xTest, yTest):
    model.eval()
    testDataset = torch.utils.data.TensorDataset(xTest, yTest)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=64, shuffle=False)
    criterion = nn.MSELoss()
    testLoss = 0.0
    predictions = []
    with torch.no_grad():
        for inputs, targets in testLoader:
            testOutputs = model(inputs)
            testLoss += criterion(testOutputs, targets).item()
            predictions.append(testOutputs)
    
    testLoss /= len(testLoader)
    print(f'Test Loss: {testLoss:.4f}')
    return torch.cat(predictions)

def plotPredictions(true_values, predicted_values, title='True vs Predicted Values'):
    if isinstance(true_values, torch.Tensor):
        true_values = true_values.detach().numpy()
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.detach().numpy()

    plt.figure(figsize=(10,6))
    plt.plot(true_values, label='True Values')
    plt.plot(predicted_values, label='Predicted Values', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()