import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt    

# class LSTMModel(nn.Module):
#     def __init__(self, inputFeatures, hiddenFeatures, layerDimension, outputDimension):
#         super(LSTMModel, self).__init__()
#         self.hiddenFeatures = hiddenFeatures
#         self.layerDimension = layerDimension   
#         self.lstm = nn.LSTM(inputFeatures, hiddenFeatures, layerDimension, batch_first=True)
#         self.fc1 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64, outputDimension)

#     def forward(self, x):
#         x, (h, c) = self.lstm(x)
#         x = x[:, -1, :] 
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         return x
    
class LSTMWrapper(nn.Module):
    def __init__(self, inputFeatures, hiddenFeatures, layerDimension, dropout_p, bidirectional):
        super(LSTMWrapper, self).__init__()
        self.lstm = nn.LSTM(inputFeatures, hiddenFeatures, layerDimension, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x) 
        return x

class LSTMModel(nn.Module):
    def __init__(self, inputFeatures, hiddenFeatures, layerDimension, outputDimension, dropout_p=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.model = nn.Sequential(
            LSTMWrapper(inputFeatures, hiddenFeatures, layerDimension, dropout_p, bidirectional),
            nn.ReLU(),
            nn.Linear(hiddenFeatures, outputDimension)
        )

    def forward(self, x):
        x = x[:, -1, :] 
        x = self.model(x)
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
    for i in range(seqLength, len(data)):
        x = data[i-seqLength:i,0]
        y = data[i,0]
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    xs = np.reshape(xs, (xs.shape[0], xs.shape[1], 1))
    ys = np.reshape(ys, (ys.shape[0], 1))
    return xs, ys

def trainModel(model, xTrain, yTrain, xVal, yVal, numEpochs, learningRate, batchSize, critery, optimizerAlgorithm, weight_decay=0.0, grad_clip=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print( 'Device:', device)
    model = model.to(device)  

    xTrain = torch.tensor(xTrain, dtype=torch.float32)
    yTrain = torch.tensor(yTrain, dtype=torch.float32)
    xVal = torch.tensor(xVal, dtype=torch.float32)
    yVal = torch.tensor(yVal, dtype=torch.float32)
    xTrain, yTrain = xTrain.to(device), yTrain.to(device) 
    xVal, yVal = xVal.to(device), yVal.to(device) 


    criterion = critery 
    optimizer = optimizerAlgorithm(model.parameters(), lr=learningRate, weight_decay=weight_decay)

    trainDataset = torch.utils.data.TensorDataset(xTrain, yTrain)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=False)
    
    valDataset = torch.utils.data.TensorDataset(xVal, yVal)
    valLoader = torch.utils.data.DataLoader(dataset=valDataset, batch_size=batchSize, shuffle=False)
    
    for epoch in range(numEpochs):
        model.train()
        trainLoss = 0.0
        for i, (inputs, targets) in enumerate(trainLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


            optimizer.step()
            trainLoss += loss.item()

        trainLoss /= len(trainLoader)

        model.eval()
        valLoss = 0.0
        with torch.no_grad():
            for inputs, targets in valLoader:
                inputs, targets = inputs.to(device), targets.to(device)
                valOutputs = model(inputs)
                valLoss += criterion(valOutputs, targets).item()

        valLoss /= len(valLoader)

        print(f'Epoch [{epoch+1}/{numEpochs}], Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}')
    return model

def evaluateModel(model, xTest, yTest):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # print( 'Device:', device)
    model = model.to(device)  
    model.eval()
    xTest = torch.tensor(xTest, dtype=torch.float32)
    yTest = torch.tensor(yTest, dtype=torch.float32)
    xTest, yTest = xTest.to(device), yTest.to(device)
    testDataset = torch.utils.data.TensorDataset(xTest, yTest)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=64, shuffle=False)
    criterion = nn.MSELoss()
    testLoss = 0.0
    predictions = []
    with torch.no_grad():
        for inputs, targets in testLoader:
            inputs, targets = inputs.to(device), targets.to(device)
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