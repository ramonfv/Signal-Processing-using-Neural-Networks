import torch
from torch import nn
from models import MLP
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
# Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
# model = MLP(input_layer=2, hidden_layer=2, output_layer=1, 
#                         learning_rate=0.1, epochs=1000, max_error=0.1)

# model.train(X, Y)

# predictions = model.predict(X)
# print(predictions)
# model.evaluate(X, Y)


# aproxime uma função seno com a MLP
numeroAmostras = 500
xTrain = torch.rand(numeroAmostras)
xTrain, _= torch.sort(xTrain)
yTrain = torch.sin(2 * torch.pi * xTrain)
# plt.plot(xTrain, yTrain)
# plt.show()


X_train, X_val, y_train, y_val = train_test_split(xTrain, yTrain, test_size=0.2)

print(f'xtrain shape: {X_train.shape}\n')
print(f'ytrain shape: {y_train.shape}\n')
print(f'xval shape: {X_val.shape}\n')
print(f'yval shape: {y_val.shape}')

model = MLP(input_layer=1, hidden_layer=5, output_layer=1, learning_rate=0.3, epochs=5000, max_error=0.1)


model.train(X_train, y_train)


predictions = model.predict(X_val)
model.evaluate(y_val, predictions)
    
