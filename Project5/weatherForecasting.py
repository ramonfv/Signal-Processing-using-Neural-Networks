from Utils.loadDatabase import ReadData
from sklearn.model_selection import train_test_split
from Utils.models import LSTMModel, prepareSequences, trainModel, evaluateModel, plotPredictions
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
import pandas as pd


## 1 - Baixe o dataset Jena Climate, dispon ́ıvel em: https://www.kaggle.com/datasets/mnassrib/jena-climate.
data = ReadData()

df = data.loadDataset("Utils/temp_extracted/jena_climate_2009_2016.csv")

## 2 - Faça a indexação dos dados temporalmente e considere somente a variável Temperatura.  ́

indexdData = data.tempIndex(df, 'Date Time', 'T (degC)')


## 3 - Particione a base de dados sendo 80% dos dados para treinar, 15% para validar e 5% para testar.
# Lembre-se de manter a sequencia temporal, ou seja, configure shuffle = 0.

train, setTest = train_test_split(indexdData['T (degC)'].values, test_size=0.2, random_state=42, shuffle=False)
validation, test = train_test_split(setTest, test_size=0.25, random_state=42, shuffle=False)

## 4 - Implemente uma rede recorrente do tipo LSTM ou GRU capaz de receber a serie temporal da variável 
# temperatura e prever a temperatura da proxima hora.  ́

scaler = MinMaxScaler(feature_range=(-1, 1))
trainNormalized = scaler.fit_transform(train.reshape(-1, 1)).flatten().reshape(-1, 1)
validationNormalized = scaler.transform(validation.reshape(-1, 1)).flatten().reshape(-1, 1)
testNormalized = scaler.transform(test.reshape(-1, 1)).flatten().reshape(-1, 1)

seq_length = 6 
xTrain, yTrain = prepareSequences(trainNormalized, seq_length)
xVal, yVal = prepareSequences(validationNormalized, seq_length)
xTest, yTest = prepareSequences(testNormalized, seq_length)

input_features = 1
hidden_features = 64
layer_dimension = 2
output_dimension = 1
num_epochs = 4
learning_rate = 0.001
batch_size = 32  
critery = nn.MSELoss()
optimizerAlgorithm = torch.optim.Adam
bidirectional = False
dropout=0.0





model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension, dropout, bidirectional)
print(f'Formato da entrada de treinamento: {xTrain.shape}')
# trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size, critery, optimizerAlgorithm)

# torch.save(trainedModel.state_dict(), 'trainedLSTMBaseModel.pth')


model.load_state_dict(torch.load('trainedLSTMBaseModel.pth'))
trainedModel = model.eval()

predicted = evaluateModel(trainedModel, xTest, yTest)
predicted = predicted.cpu().numpy()

predicted = scaler.inverse_transform(predicted)
# preditedFuture = predicted[xTest.shape[0]-1]
predicted = predicted[:xTest.shape[0]-1]
predicted  = predicted[:,0]

trueValues = indexdData['T (degC)'].values[-len(predicted):]

plotPredictions(trueValues[-seq_length:], predicted[-seq_length:], title='Previsão de Temperatura para a próxima hora')

## 5 - Varie os parametros da rede e verifique quais impactos são causados no erro de predição 
# temperatura na base de teste.

hidden_features = 128
layer_dimension = 2
# aumentando o número de camadas e diminuindo a acurácia do modelo
dropout=0.0
# aumentando o dropout(a partir de 10%) diominu a acurácia do modelo
num_epochs = 5
critery = nn.MSELoss()
optimizerAlgorithm = torch.optim.Adagrad
weight_decay = 0.001
bidirectional = False
grad_clip = 3.0


model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension, dropout, bidirectional)

# trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size, critery, optimizerAlgorithm, weight_decay, grad_clip)
# torch.save(trainedModel.state_dict(), 'trainedLSTMChangeParameters.pth')

model.load_state_dict(torch.load('trainedLSTMChangeParameters.pth'))
trainedModel = model.eval()

predicted = evaluateModel(trainedModel, xTest, yTest)
predicted = predicted.cpu().numpy()
predicted = scaler.inverse_transform(predicted)
predicted = predicted[:xTest.shape[0]-1]
predicted  = predicted[:,0]
trueValues = indexdData['T (degC)'].values[-len(predicted):]
plotPredictions(trueValues[-seq_length:], predicted[-seq_length:], title='Previsão de Temperatura para a próxima hora com parâmetros alterados')


## 6 - Altere o janelamento de amostras passadas usadas para a predição. O que acontece com a precisão
# da predição na base de teste

seq_length = 432 
xTrain, yTrain = prepareSequences(trainNormalized, seq_length)
xVal, yVal = prepareSequences(validationNormalized, seq_length)
xTest, yTest = prepareSequences(testNormalized, seq_length)

input_features = 1
hidden_features = 64
layer_dimension = 2
output_dimension = 1
num_epochs = 4
learning_rate = 0.001
batch_size = 32  
critery = nn.MSELoss()
optimizerAlgorithm = torch.optim.Adam
bidirectional = False
dropout=0.0

model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension, dropout, bidirectional)

# trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size, critery, optimizerAlgorithm)
# torch.save(trainedModel.state_dict(), 'trainedLSTM72Hours.pth')

model.load_state_dict(torch.load('trainedLSTM72Hours.pth'))
trainedModel = model.eval()

predicted = evaluateModel(trainedModel, xTest, yTest)
predicted = predicted.cpu().numpy()

predicted = scaler.inverse_transform(predicted)
predicted = predicted[:xTest.shape[0]-1]
predicted  = predicted[:,0]

trueValues = indexdData['T (degC)'].values[-len(predicted):]

plotPredictions(trueValues[-seq_length:], predicted[-seq_length:], title='Previsão de Temperatura para as próximas 72 horas')

## 7 - Considere somente os dados dos anos de 2015 em diante no treinamento do modelo. Como o
# tamanho da base de dados afeta a predic ̧ao? Com base nos resultados como você interpreta a
# relação do passado histórico com a predição do valor atual da ação?

indexdData.index = pd.to_datetime(indexdData.index, format='%d.%m.%Y %H:%M:%S')

dataFrom2015 = indexdData.loc['01-01-2015 00:00:00':]



train, setTest = train_test_split(dataFrom2015['T (degC)'].values, test_size=0.2, random_state=42, shuffle=False)
validation, test = train_test_split(setTest, test_size=0.25, random_state=42, shuffle=False)


scaler = MinMaxScaler(feature_range=(-1, 1))
trainNormalized = scaler.fit_transform(train.reshape(-1, 1)).flatten().reshape(-1, 1)
validationNormalized = scaler.transform(validation.reshape(-1, 1)).flatten().reshape(-1, 1)
testNormalized = scaler.transform(test.reshape(-1, 1)).flatten().reshape(-1, 1)

seq_length = 6
xTrain, yTrain = prepareSequences(trainNormalized, seq_length)
xVal, yVal = prepareSequences(validationNormalized, seq_length)
xTest, yTest = prepareSequences(testNormalized, seq_length)

input_features = 1
hidden_features = 64
layer_dimension = 2
output_dimension = 1
num_epochs = 4
learning_rate = 0.001
batch_size = 32  
critery = nn.MSELoss()
optimizerAlgorithm = torch.optim.Adam
bidirectional = False
dropout=0.0

model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension, dropout, bidirectional)

# trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size, critery, optimizerAlgorithm)
# torch.save(trainedModel.state_dict(), 'trainedLSTMModel2015Data.pth')

model.load_state_dict(torch.load('trainedLSTMModel2015Data.pth'))
trainedModel = model.eval()

predicted = evaluateModel(trainedModel, xTest, yTest)
predicted = predicted.cpu().numpy()

predicted = scaler.inverse_transform(predicted)
predicted = predicted[:xTest.shape[0]-1]
predicted  = predicted[:,0]

trueValues = indexdData['T (degC)'].values[-len(predicted):]

plotPredictions(trueValues[-seq_length:], predicted[-seq_length:], title='Previsão de Temperatura para a próxima hora, considerando os dados a partir de 2015')

# loss referência: 0.000
# loss a partir de: 0.001

# O tamanho da base de dados impactou pouquissímo na predição, o modelo se manteve com a acurácia muito próxima da base de dados completa.
# Por outro lado, ao aumentar o tamanho da janela de contexto a acurácia do modelo aumenta consideravelmente, o que indica que o passado histórico
# é um fator relevante para a predição do valor atual da temperatura.

## 8 - Altere o codigo para que ele seja capaz de realizar a predição da temperatura para um período maior,
# por exemplo, o valor da temperatura prevista para as proximas 24 horas. 

indexdData = data.tempIndex(df, 'Date Time', 'T (degC)')

scaler = MinMaxScaler(feature_range=(-1, 1))
trainNormalized = scaler.fit_transform(train.reshape(-1, 1)).flatten().reshape(-1, 1)
validationNormalized = scaler.transform(validation.reshape(-1, 1)).flatten().reshape(-1, 1)
testNormalized = scaler.transform(test.reshape(-1, 1)).flatten().reshape(-1, 1)

train, setTest = train_test_split(indexdData['T (degC)'].values, test_size=0.2, random_state=42, shuffle=False)
validation, test = train_test_split(setTest, test_size=0.25, random_state=42, shuffle=False)

seq_length = 1008
xTrain, yTrain = prepareSequences(trainNormalized, seq_length)
xVal, yVal = prepareSequences(validationNormalized, seq_length)
xTest, yTest = prepareSequences(testNormalized, seq_length)


input_features = 1
hidden_features = 64
layer_dimension = 2
output_dimension = 1
num_epochs = 4
learning_rate = 0.001
batch_size = 32  
critery = nn.MSELoss()
optimizerAlgorithm = torch.optim.Adam
bidirectional = False
dropout=0.0

model = LSTMModel(input_features, hidden_features, layer_dimension, output_dimension, dropout, bidirectional)

# trainedModel = trainModel(model, xTrain, yTrain, xVal, yVal, num_epochs, learning_rate, batch_size, critery, optimizerAlgorithm)

# torch.save(trainedModel.state_dict(), 'trainedLSTMModel24Hours.pth')

model.load_state_dict(torch.load('trainedLSTMModel24Hours.pth'))
trainedModel = model.eval()

predicted = evaluateModel(trainedModel, xTest, yTest)
predicted = predicted.cpu().numpy()
predicted = scaler.inverse_transform(predicted)
predicted = predicted[:xTest.shape[0]-1]
predicted  = predicted[:,0]

trueValues = indexdData['T (degC)'].values[-len(predicted):]

plotPredictions(trueValues[-seq_length:], predicted[-seq_length:], title='Previsão de Temperatura para as próximas semana')


