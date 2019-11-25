import numpy.linalg as la
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# func_logistic_regression_two_class
def func_lr_2c(beta, data, reg, samples):
    X = data['X'][samples, :]
    Y = data['Y'][samples]
    n = Y.shape[0]
    loss = (1./n) * np.sum(1+np.exp(-X.dot(beta)*Y))
    regularizer = reg/2. * la.norm(beta) ** 2
    return loss, regularizer


# grad_logistic_regression_two_class
def grad_lr_2c(beta, data, reg, samples):
    X = data['X'][samples, :]
    Y = data['Y'][samples]
    n = Y.shape[0]
    tmp = np.exp(-X.dot(beta)*Y)
    weight = tmp / (1+tmp)
    grad_loss = (1./n) * (X.T.dot(-Y*weight))
    grad_reg = reg * beta
    return grad_loss, grad_reg
