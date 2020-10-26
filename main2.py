import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.losses import MeanSquaredError, MeanAbsoluteError, \
            MeanAbsolutePercentageError, MeanSquaredLogarithmicError
from sklearn.preprocessing import MinMaxScaler

import json
import sys

from PSO import PSO
from MVO import MVO
import MVO_ENN
import PSO_ENN
from Genetic import GA
from Tools2 import *
from Model import Model

np.random.seed(2020)



def data(f):
    path = f'./Data/{f}.csv'
    data = pd.read_csv(path)
    data = data['Open']
    return data

# ********************
# MeanSquaredError class
# RootMeanSquareErro class
# MeanAbsoluteError class
# MeanSquaredLogarithmicError class
# *********************


def main():
    table = 'Table2'
    stack = 'Stack2'
    gas = 'SO2'
    NN = 'LSTM'
    OP = 'MVO'
    data = pd.read_excel(stack + '.xlsx')
    # data = data.set_index(data.iloc[:, 0])
    # data = data.iloc[:, 1:]
    # data = data.dropna()
    # data = data.iloc[1:]
    # data.to_excel('Stack2.xlsx')

    W_S = data['W_S']
    T = data['T']
    data = data[gas]
    # 250 490 737 985

    may = data[:250]
    june = data[250:490]
    july = data[490:737]
    agust = data[737:985]
    september = data[985:]

    may_w_s = W_S[:250]
    june_w_s = W_S[250:490]
    july_w_s = W_S[490:737]
    agust_w_s = W_S[737:985]
    september_w_s = W_S[985:]

    may_t = T[:250]
    june_t = T[250:490]
    july_t = T[490:737]
    agust_t = T[737:985]
    september_t = T[985:]

    d = [may, june, july, agust, september]
    d_w_s = [may_w_s, june_w_s, july_w_s, agust_w_s, september_w_s]
    d_t = [may_t, june_t, july_t, agust_t, september_t]

    dd = ['may', 'june', 'july', 'agust', 'september']

    BS = 3
    TS = None

    p = dict()
    pp = dict()
    # for i in range(5):
    #     dnn = DNN(d[i], BS, TS)
    #     rmse, mae, mape = dnn.svr()
    #     p[dd[i]] = [rmse, mae, mape]
    # pp = pd.DataFrame(p)
    # pp.to_excel('2so.xlsx')
    for i in range(5):
        X = d[i]

        X, y = prepare_data_window(X, BS, d_w_s[i], d_t[i])
        y = np.array(y)
        y = y.reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(X)
        X = scaler_X.transform(X)

        scaler_y = MinMaxScaler()
        scaler_y = scaler_y.fit(y)
        y = scaler_y.transform(y)

        y = y.reshape(-1, 1)

        # LSTM************************
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, shuffle=False)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        y_train = np.array(y_train)
        y_test = np.array(y_test)

        y_train = y_train.flatten()
        y_test = y_test.flatten()


        input_shape = X_train.shape
        model = Model().get_lstm(input_shape)
        Wb = get_weights_and_biases(model)
        Wb_flatten = flatten_weights_and_biases(Wb)
        dimensions = len(Wb_flatten)

        # # PSO
        # pso = PSO(model, dimensions, X_train, y_train, None,
        #                 init_weights=None, n_iteration=50)
        # cost, pos = pso.PSO()

        # MVO
        mvo = MVO(model, dimensions, X_train, y_train)
        cost, pos = mvo.MVO()

        model = Model().get_lstm(input_shape)
        Wb_model = unflatten_weights_and_biases(Wb, pos)
        model = put_weights_and_biases(model, Wb_model)
        y_pred = model.predict(X_test)
        # # mse = MeanSquaredError()
        # # loss = mse(y_test, y_pred).numpy()
        #
        # # with open(f'./Results/{file_name}', 'w') as f:
        # #     f.write(str(loss))
        # # LSTM--------------------------------

        # # ENN+++++++++++++++++++++++++++++++++
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, shuffle=False)
        # y_train = y_train.reshape(-1, 1)
        #
        # y_train = np.array(y_train)
        # y_test = np.array(y_test)
        #
        # # y_train = y_train.flatten()
        # # y_test = y_test.flatten()
        #
        # net = Model().get_elman(3)
        # w, s = get_elman_weights(net)
        #
        # dimensions = len(w)
        #
        # # error = net.train(X_train, y_train, epochs=500, show=100, goal=0.01)
        #
        # # PSO
        # pso_elman = PSO_ENN.PSO(net, dimensions, X_train, y_train, None,
        #           init_weights=None, n_iteration=50)
        # cost, pos = pso_elman.PSO()
        #
        # # # MVO
        # # mvo = MVO_ENN.MVO(net, dimensions, X_train, y_train)
        # # cost, pos = mvo.MVO()
        #
        # net = set_elman_weights(net, pos)
        #
        # y_pred = net.sim(X_test)
        #
        # # # model = Model().get_lstm(input_shape)
        # # # Wb_model = unflatten_weights_and_biases(Wb, pos)
        # # # model = put_weights_and_biases(model, Wb_model)
        # # # y_pred = model.predict(X_test)
        #
        # # ENN---------------------------------

        mse = MeanSquaredError()
        loss_mse = mse(y_test, y_pred).numpy()

        loss_rmse = np.sqrt(loss_mse)

        mae = MeanAbsoluteError()
        loss_mae = mae(y_test, y_pred).numpy()

        mape = MeanAbsolutePercentageError()
        loss_mape = mape(y_test, y_pred).numpy()

        p[dd[i]] = [loss_rmse, loss_mae, loss_mape]

        file_name_real = f'{stack}_{gas}_{dd[i]}_{NN}_{OP}_real.xlsx'
        file_name_pred = f'{stack}_{gas}_{dd[i]}_{NN}_{OP}_pred.xlsx'

        pp[file_name_real] = y_test
        pp[file_name_pred] = y_pred

        # plt.plot(y_pred)
        # plt.plot(y_test)
        # plt.show()

    p2 = pd.DataFrame(p)
    p2.to_excel('2so.xlsx')

    pp2 = pd.DataFrame.from_dict(pp, orient='index')
    pp2 = pp2.transpose()
    pp2.to_excel(f'{table}_{stack}_{gas}_{NN}_{OP}.xlsx')


    # with open(f'./Results/{file_name}', 'w') as f:
    #     f.write(f'MSE: {str(loss_mse)}')
    #     f.write('\n')
    #     f.write(f'RMSE: {str(loss_rmse)}')
    #     f.write('\n')
    #     f.write(f'MAE: {str(loss_mae)}')
    #     f.write('\n')
    #     f.write(f'MAPE: {str(loss_mape)}')
    #     f.write('\n')
    #     f.write(f'MSLE: {str(loss_msle)}')
    #     f.write('\n')


def pdf_creator(image_path, text):
    from fpdf import FPDF
    pdf = FPDF()
    for i in range(len(image_path)):
        pdf.add_page()
        pdf.image(image_path[i], x=10, y=8, w=160)
        pdf.set_font('Arial', size=12)
        pdf.ln(120)
        pdf.cell(200, 10, txt='{}'.format(text[i]), ln=1)
    pdf.output('results.pdf')


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # m = np.mean(y_true)
    # mape = 100 * (np.mean (np.abs(y_pred-y_true) / m))
    # mape2 = MeanAbsolutePercentageError()
    # mape2 = mape2(y_true, y_pred).numpy()
    return mape

# images = list()
# text = list()
# print(image)
# print(text)

main()

# pdf_creator(images, text)
