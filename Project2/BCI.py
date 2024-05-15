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


mat = sp.loadmat(load_env('MAT_FILE_PATH'), verify_compressed_data_integrity=False)
data = np.array(mat['data'])
frequencyStimulus = [2, 7] # frequencias 10 e 15
electrode = [61]
inputData, labels = get_data(data, frequencyStimulus, electrode)
inputDataResampled = np.reshape(inputData, (12,1250))

classe1 = inputDataResampled[0:6, :]
classe2 = inputDataResampled[6:12, :]

# freqD = freqDomain(inputData, 0, 3, 250)
# freq, _ = freqD.variables(inputData, 250)
# print(freq)


classe1 = inputDataResampled[0:6, :]
classe2 = inputDataResampled[6:12, :]

graphs.scatterPlot2Labels(classe1.T, classe2.T, 61, '10Hz', '15Hz')

# train_data, valid_data, train_labels, valid_labels = train_test_split(inputDataResampled, labels, test_size=0.33)
# valid_labels = valid_labels.flatten()

# learningRate = [0.05, 0.1, 0.2, 0.5, 0.7]
# epochs = [10, 20, 15, 5, 8]

# for i in range(len(learningRate)):
#     train_data, valid_data, train_labels, valid_labels = train_test_split(inputDataResampled, labels, test_size=0.33)
#     valid_labels = valid_labels.flatten()
#     model = Perceptron(train_data, train_labels, learningRate[i] , epochs[i])
#     weights = model.train_perceptron()
#     predict = model.predict(valid_data, weights) 
#     accuracy = model.evaluate(predict, valid_labels)
#     print("Epoch:", epochs[i], "- Learning Rate:", learningRate[i])
#     print("Accuracy:", accuracy)
    # model.plot_3d(valid_data, valid_labels, predict)

# Melhor resultado


train_data, valid_data, train_labels, valid_labels = train_test_split(inputDataResampled, labels, test_size=0.33)
valid_labels = valid_labels.flatten()


model = Perceptron(train_data, train_labels, 0.7, 20)
weights = model.train_perceptron()
predict = model.predict(valid_data, weights) 
accuracy = model.evaluate(predict, valid_labels)
print("Accuracy:", accuracy)
model.plot_3d(valid_data, valid_labels, predict)


graphs.confusionMatrix(valid_labels, predict, [1, -1])

freqD = freqDomain(inputData, 0, 3, 250)
freqVector = freqD.variables(inputData, 250, 0, 3)

freqClass1Doamin, mag1 = freqD.computeFFT(inputData, 0, 3)
freqClass2Doamin, mag2 = freqD.computeFFT(inputData, 1, 3)

concatenated_freq_domain = np.vstack((freqClass1Doamin, freqClass2Doamin))
freqD.fftPlot(inputData, mag1, 0, 250, 3)

featureMatrx = freqD.featureExtraction(concatenated_freq_domain, freqVector, [10, 15], 2, 3)

featureMatrx = np.array(featureMatrx)
labelsfft = np.array([1, -1])

labelsfft = np.array(labelsfft).flatten()

train_fft, valid_fft, train_fft_labels, valid_fft_labels = train_test_split(featureMatrx, labelsfft, test_size=0.33)

modelfft = Perceptron(train_fft, train_fft_labels, 0.1 , 20)
weightsfft = modelfft.train_perceptron()

predictft = modelfft.predict(valid_fft, weightsfft) 
accuracyfft = modelfft.evaluate(predictft, valid_fft_labels)
print("Acurácia com dados da FFT:", accuracyfft)
print("Rótulos verdadeiros com dados da FFT:", valid_fft_labels)
print("Previsões com dados da FFT:", predictft)
