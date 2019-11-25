import os
import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.datasets import mnist

class DataCancer(object):
    @staticmethod
    def load_data():
        # save mnist data
        data_path = "../data/cancer/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            (x,y) = load_breast_cancer(return_X_y=True)
            y = y*2 - 1
            num_train = 480
            x_train, y_train, x_test, y_test = x[:num_train,:],y[:num_train],x[num_train:,:],y[num_train:],
            data = data_dictionary(x_train, y_train, x_test, y_test)
            np.save("../data/cancer/cancer.npy",data)
        else:
            data = np.load("../data/cancer/cancer.npy", allow_pickle=True)[()]
            train = data['train']
            test = data['test']
            data = {'train': train, 'test': test}
        scaler = MinMaxScaler()
        scaler.fit(data['train']['X'])
        data['train']['X']=scaler.transform(data['train']['X'])
        data['test']['X'] = scaler.transform(data['test']['X'])
        return data

class DataMNIST(object):
    @staticmethod
    def load_data():
        # save mnist data
        data_path = "../data/mnist/"
        data, train, test = {}, {}, {}
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            idx_tr = (y_train == 7) | (y_train == 9)
            idx_te = (y_test == 7) | (y_test == 9)
            x_train, y_train = x_train[idx_tr], y_train[idx_tr].astype(int)
            x_test, y_test = x_test[idx_te], y_test[idx_te].astype(int)
            y_train = y_train - 8
            y_test = y_test - 8
            x_train = x_train.reshape(x_train.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)
            x_train = x_train.astype('float32') / 255
            x_test = x_test.astype('float32') / 255
            data = data_dictionary(x_train, y_train, x_test, y_test)
            np.save("../data/mnist/mnist.npy",data)
        else:
            data = np.load("../data/mnist/mnist.npy", allow_pickle=True)[()]
            train = data['train']
            test = data['test']
            data = {'train':train, 'test':test}
        return data

def data_dictionary(x_train, y_train, x_test, y_test):
    data, train, test = {}, {}, {}
    train['X'] = x_train
    train['Y'] = y_train
    train['n'], train['d'] = x_train.shape
    train['C'] = len(set(train['Y']))

    test['X'] = x_test
    test['Y'] = y_test
    test['n'], test['d'] = x_test.shape
    test['C'] = len(set(test['Y']))
    data['train'] = train
    data['test'] = test
    return data