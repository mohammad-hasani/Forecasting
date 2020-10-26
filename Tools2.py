import numpy as np
import pandas as pd

import functools
import operator
import Model


def get_weights_and_biases(model):
    weights = model.get_weights()
    weights = np.array(weights)
    return weights


def get_elman_weights(net):
    w = list()
    s = list()
    for i in range(len(net.layers)):
        layer = net.layers[i].np['w']
        s.append(layer.shape)
        w.extend(layer.flatten())
    return w, s

def set_elman_weights(net, w):
    _, s = get_elman_weights(net)
    counter = 0
    for i in range(len(s)):
        start = 0 if i == 0 else s[i-1][0] * s[i-1][1]
        new_w = w[start + counter: start + counter + s[i][0] * s[i][1]]
        counter += start
        new_w = np.array(new_w)
        new_w = new_w.reshape(s[i])
        net.layers[i].np['w'] = new_w

    return net


def flatten_weights_and_biases(Wb):
    vec = list()
    for i in range(Wb.shape[0]):
        tmp = Wb[i].flatten()
        vec.extend(tmp)
    return vec


def unflatten_weights_and_biases(Wb, Wb_flatten):
    mat = list()
    last_index = 0
    for i in range(Wb.shape[0]):
        shape = Wb[i].shape
        number = functools.reduce(operator.mul, shape)

        l = Wb_flatten[last_index:last_index + number]
        l = np.array(l)
        l = l.reshape(shape)
        mat.append(l)

        last_index = number

    return mat


def put_weights_and_biases(model, Wb):
    model.set_weights(Wb)
    return model


def data_prepration(data):
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    return data


def prepare_data_window(X, WINDOW_SIZE, W_S=None, T=None):
        series = pd.Series(X)
        series_s = series.copy()

        for i in range(WINDOW_SIZE):
            series = pd.concat([series, series_s.shift(-(i+1))], axis=1)
        series.dropna(axis=0, inplace=True)
        series.columns = np.arange(WINDOW_SIZE + 1)

        X_new = pd.DataFrame()
        for i in range(WINDOW_SIZE):
            X_new[i] = series[i]
        Y_new = series[WINDOW_SIZE]
        if T is not None:

            X_new = X_new.reset_index(drop=True)

            T = T.reset_index(drop=True)
            T = T.drop([0, 1, 2])
            T = T.reset_index(drop=True)
            X_new = X_new.join(T)

            W_S = W_S.reset_index(drop=True)
            W_S = W_S.drop([0, 1, 2])
            W_S = W_S.reset_index(drop=True)
            X_new = X_new.join(W_S)

        # print(X_new)

        return X_new, Y_new


