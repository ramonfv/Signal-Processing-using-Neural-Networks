from Utils.loadDatabase import ReadData
from sklearn.model_selection import train_test_split
from Utils.models import LSTMModel, prepareSequences, trainModel, evaluateModel, plotPredictions
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


# print(torch.cuda.is_available())

data = ReadData()

df = data.loadDataset("Utils/temp_extracted/jena_climate_2009_2016.csv")

indexdData = data.tempIndex(df, 'Date Time', 'T (degC)')

# print(indexdData.describe())

train, setTest = train_test_split(indexdData['T (degC)'].values, test_size=0.2, random_state=42, shuffle=False)

validation, test = train_test_split(setTest, test_size=0.25, random_state=42, shuffle=False)

scaler = MinMaxScaler(feature_range=(-1, 1))

trainNormalized = scaler.fit_transform(train.reshape(-1, 1)).flatten().reshape(-1, 1)
validationNormalized = scaler.transform(validation.reshape(-1, 1)).flatten().reshape(-1, 1)
testNormalized = scaler.transform(test.reshape(-1, 1)).flatten().reshape(-1, 1)


seq_length = 6 
xTrain, yTrain = prepareSequences(trainNormalized, seq_length)
xVal, yVal = prepareSequences(validationNormalized, seq_length)
xTest, yTest = prepareSequences(testNormalized, seq_length)


# print(xTrain.shape, yTrain.shape)
# print(xVal.shape, yVal.shape)
# print(xTest.shape, yTest.shape)


xTrain = torch.tensor(xTrain, dtype=torch.float32)
yTrain = torch.tensor(yTrain, dtype=torch.float32)
xVal = torch.tensor(xVal, dtype=torch.float32)
yVal = torch.tensor(yVal, dtype=torch.float32)
xTest = torch.tensor(xTest, dtype=torch.float32)
yTest = torch.tensor(yTest, dtype=torch.float32)

print(xTrain.shape, yTrain.shape, xVal.shape, yVal.shape, xTest.shape, yTest.shape)

input_features = 1
hidden_features = 64
layer_dimension = 1
output_dimension = 1
num_epochs = 4
learning_rate = 0.001
batch_size = 32  
model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension)
print(f'Formato da entrada de treinamento: {xTrain.shape}')
trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size)

predicted = evaluateModel(trainedModel, xTest, yTest)

print(predicted.shape)

predicted = scaler.inverse_transform(predicted)
predicted = predicted.flatten()
predicted = predicted.reshape(-1, 1)

plotPredictions(indexdData['T (degC)'].values[-len(predicted):], predicted, title='True vs Predicted Temperature')