from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt


def get_data():

    url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20181101"

    r = requests.get(url)
    data = r.text

    soup = BeautifulSoup(data)
    table = soup.find('table')
    tbody = table.find('tbody')
    tr_all = tbody.find_all('tr', attrs={'class': 'text-right'})
    info = list()
    header = (['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'])

    for i in tr_all:
        tmp_data = i.find_all('td')
        tmp_list = list()
        for j in tmp_data:
            tmp_list.append(j.renderContents().decode())
        info.append(tmp_list)

    info = pd.DataFrame(info, columns=header)
    for i, v in enumerate(header):
        info[v] = info[v].values[::-1]
        if i >= 5:
            info[v] = info[v].str.replace(',', '')
            info[v] = info[v].str.replace('-', '0')
        try:
            info[v] = info[v].astype('float32')
        except Exception as e:
            pass

    info.to_csv('data.csv')


def plot_data():
    data = pd.read_csv('data.csv')
    date = data['Date']
    data_open = data['Volume']
    plt.plot(date, data_open)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.show()


def read_data():
    path = './Data/'
    data = list()
    for a1, a2, a3 in os.walk(path):
        for i in a3:
            tmp_path = path + str(i)
            tmp_data = pd.read_csv(tmp_path)
            data.append([i, tmp_data])
    return data


def read_one_data(path):
    data = pd.read_csv(path)
    return data


def save_info(y, y_predict, name, mape_error, window_size, path, bs, ts):
    # colors = ['b', 'r']
    # plt.plot(y, colors[0])
    # plt.plot(y_predict, colors[1])
    # plt.savefig(path + name + '/' + name + str(window_size) + '.jpg')

    with open(path + '/' + name + '.txt', 'a') as f:
        msg = """
            %d, %d, %s
        """ % (int(bs), int(ts), mape_error)
        f.write(msg)


def show_single_plot(X, name):
    plt.plot(X)
    plt.savefig('./new/' + name + '.jpg')
    plt.close()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def sort_data(path, name):
    data = list()
    with open(path + name, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) is not 0:
                data.append(line.strip())
    info = list()
    for i in range(0, len(data), 3):
        window_size = int(data[i + 1].split(':')[1].strip())
        mape = float(data[i + 2].split(':')[1].strip())
        info.append((window_size, mape))
    info.sort(key=lambda x: x[1])
    with open(path + name + ' sorted.txt', 'w') as f:
        for i in info:
            msg = "%d, %f\n" % (i[0], i[1])
            f.write(str(msg))


def get_window_sizes():
    path = './Bitcoin/Bitcoin sorted.txt'
    data = list()
    with open(path, 'r') as f:
        for i in range(10):
            window_size = f.readline().split(',')[0]
            data.append(window_size)
    return data


def recognize_data():
    for _, _, i in os.walk('/home/sina/Desktop/keynia/2'):
        for j in i:
            name = j.split('.')[0]
            data = list()
            data.append('bs,ts,mape\n')
            with open('/home/sina/Desktop/keynia/2/' + name + '.txt', 'r') as f:
                for k in f.readlines():
                    if len(k.strip()):
                        data.append(k.strip() + '\n')
            with open('/home/sina/Desktop/keynia/2/' + name + ' ts=3bs.csv', 'w') as f:
                for k in data:
                    f.write(k)


def show_plot(r, p):
    plt.plot(r, label='real')
    plt.plot(p, label='predict')
    plt.legend()
    # plt.show()
    plt.savefig('plots/ethereum_DNN_100_300.jpg')


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
