# consider the input values x = [1,4,5,8,10] and the respective output y = [3,9,11,17,21]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def FindAb(input, output):
    model = LinearRegression()
    model.fit(np.array(input).reshape(-1, 1), output)
    a = model.coef_[0]
    b = model.intercept_
    return a, b


inputs = np.array([1,4,5,8,10])
outputs = np.array([3,9,11,17,21])
