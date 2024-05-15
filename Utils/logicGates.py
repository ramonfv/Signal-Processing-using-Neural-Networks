import numpy as np

def andGate(numberInputs):
    data = np.array([[0] * numberInputs])
    labels = np.array([0])

    for i in range(2**numberInputs):
        inputs = [int(x) for x in list(bin(i)[2:].zfill(numberInputs))]
        data = np.vstack((data, inputs))
        labels = np.append(labels, 1 if sum(inputs) == numberInputs else 0)


    return data[1:], labels[1:]

def orGate(numberInputs):
    num_possibilities = 2**numberInputs
    data = np.zeros((num_possibilities, numberInputs), dtype=int)
    labels = np.zeros(num_possibilities, dtype=int)

    for i in range(num_possibilities):
        inputs = [int(x) for x in list(bin(i)[2:].zfill(numberInputs))]
        data[i] = inputs
        labels[i] = 1 if sum(inputs) > 0 else 0

    return data, labels

