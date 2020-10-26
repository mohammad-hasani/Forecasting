import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Nadam
from sklearn.model_selection import train_test_split
from tools import save_info, show_plot
from tools import mean_absolute_percentage_error
from scipy.stats.stats import pearsonr
import time
from sklearn.preprocessing import MinMaxScaler
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
import matplotlib.pyplot as plt


np.random.seed(2020)


class DNN(object):
    def __init__(self, X, bs, ts):
        self.NB_EPOCH = 200
        self.BATCH_SIZE = 1000
        self.VERBOSE = 1
        self.OPTIMIZER = Adam()
        self.N_HIDDEN = 512
        self.VALIDATION_SPLIT = 0.14
        self.INPUT_DIM = int(bs)

        self.bs = bs
        self.ts = ts

        self.WINDOW_SIZE = int(bs)

        X, Y = self.prepare_data(X)
        Y = np.array(Y)

        Y = Y.reshape(-1, 1)


        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(X)
        X = scaler_X.transform(X)

        scaler_Y = MinMaxScaler()
        scaler_Y = scaler_Y.fit(Y)
        Y = scaler_Y.transform(Y)



        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        # X = X[len(X) - ts:]
        # Y = Y[len(Y) - ts:]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.33, shuffle=False)


        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.y_train = Y_train
        self.X_test = X_test
        self.y_test = Y_test

        print(X_train.shape)

    def prepare_data(self, X):
        series = pd.Series(X)
        series_s = series.copy()

        for i in range(self.WINDOW_SIZE):
            series = pd.concat([series, series_s.shift(-(i+1))], axis=1)
        series.dropna(axis=0, inplace=True)
        series.columns = np.arange(self.WINDOW_SIZE + 1)

        X_new = pd.DataFrame()
        for i in range(self.WINDOW_SIZE):
            X_new[i] = series[i]
        Y_new = series[self.WINDOW_SIZE]

        return X_new, Y_new

    def dnn(self, path=None, name=None):
        model = Sequential()
        print(len(self.X_train))
        model.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM,)))
        model.add(Activation('relu'))
        for i in range(7):
            model.add(Dense(self.N_HIDDEN))
            model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.summary()
        model.compile(loss='mse',
                      optimizer=self.OPTIMIZER,
                      metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train,
                            epochs=self.NB_EPOCH,
                            verbose=self.VERBOSE)

        y_pred = model.predict(self.X_test)

        y_pred = y_pred.reshape(-1)

        # mae = MeanAbsoluteError()
        # error = mae(self.y_test, y_predict).numpy()
        # print(error)
        #
        # mape = MeanAbsolutePercentageError()
        # error = mape(self.y_test, y_predict).numpy()
        #
        # print(error)
        #
        plt.plot(self.y_test)
        plt.plot(y_pred)
        plt.legend(['real', 'prediction'])
        plt.savefig(name + '.png')
        plt.show()
        r = pd.DataFrame(self.y_test)
        p = pd.DataFrame(y_pred)
        r.to_excel(name + '-real.xlsx')
        p.to_excel(name + '-prediction.xlsx')


        # mape_error = mean_absolute_percentage_error(self.y_test, y_predict)

        # save_info(self.y_test, y_predict, name, mape_error, self.WINDOW_SIZE, path, self.bs, self.ts)
        mse = MeanSquaredError()
        loss_mse = mse(self.y_test, y_pred).numpy()

        loss_rmse = np.sqrt(loss_mse)

        mae = MeanAbsoluteError()
        loss_mae = mae(self.y_test, y_pred).numpy()

        mape = MeanAbsolutePercentageError()
        loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mae, loss_mape

    def lstm(self, path=None, name=None):
        trainX = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        testX = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(32, batch_input_shape=(1, trainX.shape[1], trainX.shape[2]), stateful=True))
        # model.add(Activation('tanh'))
        model.add(Dense(1))
        # model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # model.summary()

        for i in range(50):
            model.fit(trainX, self.y_train, epochs=1, batch_size=1, verbose=self.VERBOSE, shuffle=False)
            model.reset_states()

        y_pred = model.predict(testX, batch_size=1)

        y_pred = y_pred.reshape(-1)

        # mae = MeanAbsoluteError()
        # error = mae(self.y_test, y_predict).numpy()

        # mape = MeanAbsolutePercentageError()
        # error = mape(self.y_test, y_predict).numpy()

        mse = MeanSquaredError()
        loss_mse = mse(self.y_test, y_pred).numpy()

        loss_rmse = np.sqrt(loss_mse)

        mae = MeanAbsoluteError()
        loss_mae = mae(self.y_test, y_pred).numpy()

        mape = MeanAbsolutePercentageError()
        loss_mape = mape(self.y_test, y_pred).numpy()

        # msle = MeanSquaredLogarithmicError()
        # loss_msle = msle(y_test, y_pred).numpy()

        # print(error)
        #
        # plt.plot(self.y_test)
        # plt.plot(y_predict)
        # plt.show()

        # save_info(self.y_test, y_predict, name, mape_error, self.WINDOW_SIZE, path, self.bs, self.ts)
        # show_plot(self.y_test, y_predict)
        return loss_rmse, loss_mae, loss_mape

    def svr(self):
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        regr = make_pipeline(SVR(C=1.0, epsilon=0.2))
        regr.fit(self.X_train, self.y_train)

        y_pred = regr.predict(self.X_test)

        y_pred = y_pred.reshape(-1)

        mse = MeanSquaredError()
        loss_mse = mse(self.y_test, y_pred).numpy()

        loss_rmse = np.sqrt(loss_mse)

        mae = MeanAbsoluteError()
        loss_mae = mae(self.y_test, y_pred).numpy()

        mape = MeanAbsolutePercentageError()
        loss_mape = mape(self.y_test, y_pred).numpy()

        return loss_rmse, loss_mae, loss_mape

    @staticmethod
    def ENN():
        import neurolab as nl

        # Create train samples
        i1 = np.sin(np.arange(0, 20))
        i2 = np.sin(np.arange(0, 20)) * 2

        t1 = np.ones([1, 20])
        t2 = np.ones([1, 20]) * 2

        input = np.array([i1, i2, i1, i2]).reshape(20 * 4, 1)
        target = np.array([t1, t2, t1, t2]).reshape(20 * 4, 1)

        # Create network with 2 layers
        net = nl.net.newelm([[-1, 1]], [50, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
        # Set initialized functions and init
        net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
        net.init()
        # Train network
        error = net.train(input, target, epochs=500, show=100, goal=0.01)
        # Simulate network
        output = net.sim(input)
        print(output)

        # Plot result
        import pylab as pl
        pl.subplot(211)
        pl.plot(error)
        pl.xlabel('Epoch number')
        pl.ylabel('Train error (default MSE)')

        pl.subplot(212)
        pl.plot(target.reshape(80))
        pl.plot(output.reshape(80))
        pl.legend(['train target', 'net output'])
        pl.show()



        # y_pred = regr.predict(self.X_test)
        #
        # y_pred = y_pred.reshape(-1)
        #
        # mse = MeanSquaredError()
        # loss_mse = mse(self.y_test, y_pred).numpy()
        #
        # loss_rmse = np.sqrt(loss_mse)
        #
        # mae = MeanAbsoluteError()
        # loss_mae = mae(self.y_test, y_pred).numpy()
        #
        # mape = MeanAbsolutePercentageError()
        # loss_mape = mape(self.y_test, y_pred).numpy()
        #
        # return loss_rmse, loss_mae, loss_mape


#     def calculate_corr(self):
#         # corr = pearsonr(self.X.T, self.Y.T)
#         # np.savetxt('corrdata.txt', corr)
#         np.savetxt('X.txt', self.X)
#         np.savetxt('Y.txt', self.Y)

