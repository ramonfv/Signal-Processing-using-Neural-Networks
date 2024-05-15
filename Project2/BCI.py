from utils.inputData import load_env, get_data
import numpy as np
from utils.models import Perceptron
import scipy.io as sp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.frequencyDomain import freqDomain
from utils.plots import graphs


print(np.arange(0,250,1).shape)

mat = sp.loadmat(load_env('MAT_FILE_PATH'), verify_compressed_data_integrity=False)
data = np.array(mat['data'])
frequencyStimulus = [2, 7] # frequencias 10 e 15
electrode = [61]
inputData, labels = get_data(data, frequencyStimulus, electrode)
inputDataResampled = np.reshape(inputData, (12,1250))


# freqD = freqDomain(inputData, 0, 3, 250)
# freq, _ = freqD.variables(inputData, 250)
# print(freq)


classe1 = inputDataResampled[0:6, :]
classe2 = inputDataResampled[6:12, :]

graphs.scatterPlot2Labels(classe1.T, classe2.T, 61, '10Hz', '15Hz')

# train_data, valid_data, train_labels, valid_labels = train_test_split(inputDataResampled, labels, test_size=0.67)
# valid_labels = valid_labels.flatten()

# model = Perceptron(train_data, train_labels, 0.01 , 500)
# weights = model.train_perceptron()
# predict = model.predict(valid_data, weights) 
# accuracy = model.evaluate(predict, valid_labels)
# print(accuracy)
# print(valid_labels)
# print(predict)
# model.plot_3d(valid_data, valid_labels, predict)


# graphs.confusionMatrix(valid_labels, predict, [1, -1], 'Blues')

# freqDomain.computeFFT(data = inputData, frequencia = 0, amostra = 3, fs = 250)
# freqDomain.computeFFT(data = inputData, frequencia = 1, amostra = 3, fs = 250)
