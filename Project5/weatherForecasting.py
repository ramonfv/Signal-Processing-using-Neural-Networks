from Utils.loadDatabase import ReadData
from sklearn.model_selection import train_test_split
from Utils.models import LSTMModel, prepareSequences, trainModel, evaluateModel, plotPredictions
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

data = ReadData()

df = data.loadDataset("Utils/temp_extracted/jena_climate_2009_2016.csv")

indexdData = data.tempIndex(df, 'Date Time', 'T (degC)')

# print(indexdData.describe())

train, setTest = train_test_split(indexdData['T (degC)'].values, test_size=0.2, random_state=42, shuffle=False)

validation, test = train_test_split(setTest, test_size=0.25, random_state=42, shuffle=False)

scaler = MinMaxScaler(feature_range=(-1, 1))
trainNormalized = scaler.fit_transform(train.reshape(-1, 1)).flatten()
validationNormalized = scaler.transform(validation.reshape(-1, 1)).flatten()
testNormalized = scaler.transform(test.reshape(-1, 1)).flatten()

seq_length = 20  
xTrain, yTrain = prepareSequences(trainNormalized, seq_length)
xVal, yVal = prepareSequences(validationNormalized, seq_length)
xTest, yTest = prepareSequences(testNormalized, seq_length)

xTrain = torch.tensor(xTrain, dtype=torch.float32).unsqueeze(-1)
yTrain = torch.tensor(yTrain, dtype=torch.float32).unsqueeze(-1)  # Ensure yTrain has the correct shape
xVal = torch.tensor(xVal, dtype=torch.float32).unsqueeze(-1)
yVal = torch.tensor(yVal, dtype=torch.float32).unsqueeze(-1)  # Ensure yVal has the correct shape
xTest = torch.tensor(xTest, dtype=torch.float32).unsqueeze(-1)
yTest = torch.tensor(yTest, dtype=torch.float32).unsqueeze(-1)  # Ensure yTest has the correct shape

input_features = 1
hidden_features = 50
layer_dimension = 5
output_dimension = 1
num_epochs = 50
learning_rate = 0.001
batch_size = 32  # Set an appropriate batch size

model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension)

trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size)

predicted = evaluateModel(trainedModel, xTest, yTest)

plotPredictions(indexdData['T (degC)'].values[-len(predicted):], predicted, title='True vs Predicted Temperature')