from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError
import numpy as np
import neurolab as nl


class Model(object):
    def __init__(self, X_train=None, y_train=None):
        self.X_train = X_train
        self.y_train = y_train

    def get_model(self):
        pass

    def get_lstm(self,input_shape, neurons=32, layers=1):
        # X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        # input_shape = X.shape
        model = Sequential()
        for layer in range(layers):
            model.add(LSTM(neurons, input_shape=(input_shape[1], input_shape[2])))
        model.add(Dense(1))
        return model

    def get_dnn(self,input_shape, neurons=64, layers=2):
        model = Sequential()
        model.add(Dense(input_shape, input_shape=(input_shape,)))
        model.add(Activation('relu'))
        for i in range(layers):
            model.add(Dense(neurons))
            model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))

        return model

    def get_elman(self, ni):
        # Create network with 2 layers
        inputs = [[-1, 1]] * (ni + 2)
        print(inputs)
        net = nl.net.newelm(inputs, [50, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        # Set initialized functions and init
        net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.init()

        return net

    def train_model(self, model, optimizer):
        model.compile(loss='mse', optimizer=eval(f'{optimizer}()'), metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train, epochs=200)
        return model
