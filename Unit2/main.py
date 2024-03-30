from Utils.models import Perceptron
from Utils.logicGates import andGate, orGate

data, labels = andGate(3)

print(data, labels )

perceptron_and = Perceptron(data, labels, 0.05, 50)

weights_and = perceptron_and.train_perceptron()

predictions_and = perceptron_and.predict(data, weights_and)

perceptron_and.plot()