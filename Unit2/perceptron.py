# implemente atraves do algoritmo do perceptron a funcao logica AND

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
import pandas as pd

# fazendo a funcao de ativacao
def step_function(x):
    return 1 if x >= 0 else 0

# funcao do perceptron
def perceptron(x, w):
    return step_function(np.dot(x, w))

# funcao de treinamento
def train_perceptron(data, labels, epochs=50, learning_rate=0.05):
    w = np.random.rand(3)
    for _ in range(epochs):
        for i in range(len(data)):
            x = np.array([data.iloc[i,2], data.iloc[i,0], data.iloc[i,1]])
            y = perceptron(x, w)
            w += learning_rate * (labels[i] - y) * x
    return w



if __name__ == "__main__":

    iris = datasets.load_iris()
    dataIris = pd.DataFrame(iris.data)
    tagetsIris = pd.DataFrame(iris.target)
    irisColumns = pd.DataFrame(iris.feature_names)

    # 0 = setosa
    # 1 = versicolor

    dataSet = pd.DataFrame(iris.data, columns=iris.feature_names)
    dataSet['label'] = iris.target
    dataSet['species'] = dataSet['label'].replace(to_replace=[0,1,2], value=[iris.target_names[0], iris.target_names[1], iris.target_names[2]])
    inputData = dataSet[dataSet['label'] != 2]
    labels = inputData['label']
    inputData = inputData.drop(columns=['species'])
    inputData = inputData .drop(columns=['label'])
    x0 = np.ones(len(inputData))
    inputData['x0'] = x0
    
    data = inputData.drop(columns=['sepal width (cm)', 'sepal length (cm)'])

    print(data)



    
    # # dados
    # data = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    # # porta AND
    # labelsAnd = np.array([0, 0, 0, 1])
    
    #   # porta OR
    # labelsOr = np.array([0, 1, 1, 1])

    # treinando o perceptron
    w = train_perceptron(data, labels)
    print(w)
    
    
    # ponto 1: x1 = 0 --> x2 = - w0/w2
    # ponto 2: x2 = 0 --> x1 = -w0/w1
    # print 3D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inputData.iloc[:, 0], inputData.iloc[:, 1], labels , c='r', marker='o')
    x = np.linspace(0, -w[0]/w[2], 10)
    y = np.linspace(-w[0]/w[1], 0, 10)
    x, y = np.meshgrid(x, y)
    z = (w[0] + w[1] * x + w[2]) * y  
    ax.plot_surface(x, y, z)
    plt.show()


    

