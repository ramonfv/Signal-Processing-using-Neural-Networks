# from inputData import load_env, get_data
# # from Unit2.Utils.models import Perceptron
# import numpy as np
# import scipy.io as sp


# mat = sp.loadmat(load_env('MAT_FILE_PATH'), verify_compressed_data_integrity=False)
# data = np.array(mat['data'])
# frequencyStimulus = [2, 7] # frequencias 10 e 15
# electrode = [61]
# inputData, labels = get_data(data, frequencyStimulus, electrode)
# inputDataResampled = np.reshape(inputData, (12,1250))

# classe1 =  inputDataResampled[0:6, :]
# classe2 =  inputDataResampled[6:12, :]
# print(classe1.shape) # (6, 1250)
# print(classe2.shape) # (6, 1250)

# # agora, separe aleatoriamente 4 amostras de cada classe para treinar o perceptron e o restante das amostras para validar o sistema

# np.random.shuffle(classe1)
# np.random.shuffle(classe2)

# train_classe1 = classe1[0:4, :]
# train_classe2 = classe2[0:4, :]
# valid_classe1 = classe1[4:6, :]
# valid_classe2 = classe2[4:6, :]


# train_data = np.vstack((train_classe1, train_classe2))
# valid_data = np.vstack((valid_classe1, valid_classe2))

# # treinando o perceptron
# # data, labels, learning_rate, epochs=None
# model = Perceptron(train_data, labels, 0.05, 500)
# weights = model.train_perceptron()
# predict = model.predict(valid_data, weights)
# print("Predição:", predict)
# model.plot()


