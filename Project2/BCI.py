from utils.inputData import load_env, get_data
import numpy as np
from utils.models import Perceptron
import scipy.io as sp



mat = sp.loadmat(load_env('MAT_FILE_PATH'), verify_compressed_data_integrity=False)
data = np.array(mat['data'])
frequencyStimulus = [2, 7] # frequencias 10 e 15
electrode = [61]
inputData, labels = get_data(data, frequencyStimulus, electrode)
inputDataResampled = np.reshape(inputData, (12,1250))


new_labels = np.concatenate((labels[0:4], labels[6:10]))

classe1 =  inputDataResampled[0:6, :]
classe2 =  inputDataResampled[6:12, :]
print(classe1.shape) # (6, 1250)
print(classe2.shape) # (6, 1250)

np.random.shuffle(classe1)
np.random.shuffle(classe2)

train_classe1 = classe1[0:4, :]
train_classe2 = classe2[0:4, :]
valid_classe1 = classe1[4:6, :]
valid_classe2 = classe2[4:6, :]


train_data = np.vstack((train_classe1, train_classe2))
valid_data = np.vstack((valid_classe1, valid_classe2))



print(new_labels.shape)
model = Perceptron(train_data, new_labels, 0.005, 1000)
weights = model.train_perceptron()
predict = model.predict(valid_data, weights)
print("Predição:", predict)
print("real:", new_labels)
model.plot()




