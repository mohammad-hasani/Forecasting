import matplotlib.pyplot as plt
# import os
# import pandas as pd
# import numpy as np
#
# data = dict()
# for i, j, k in os.walk('./plots/'):
#     for l in k:
#         if l.startswith('Stack2') and l.endswith('.xlsx'):
#             path = i + l
#             d = pd.read_excel(path)
#             d = d.iloc[:, 1]
#             d = np.array(d)
#             data[l] = d
#
#
# print(data)
# # data = pd.DataFrame(data)
#
# data = pd.DataFrame.from_dict(data, orient='index')
# data = data.transpose()
# data.to_excel('Stack2-Result.xlsx')


def plot():
    data = pd.read_excel('Stack1-Result.xlsx')
    names = list(data.columns)
    names.sort()
    print(names)

    NO2 = names[:6]
    SO2 = names[6:12]

    print(NO2)
    print(SO2)


    counter = 0
    fig, axs = plt.subplots(2, 3)
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    for i in range(2):
        for j in range(3):
            real = data[names[counter + 1]]
            pred = data[names[counter]]
            axs[i, j].plot(real)
            axs[i, j].plot(pred)
            n = names[counter].split('.')[0].split('-')
            n = n[:3]
            n = ' '.join(n)
            axs[i, j].set_title(n)
            counter += 2
    fig.savefig('Stack1-scatter.png')


def scatter():
    data = pd.read_excel('Stack2-Result.xlsx')
    names = list(data.columns)
    names.sort()
    print(names)

    NO2 = names[:6]
    SO2 = names[6:12]

    print(NO2)
    print(SO2)

    counter = 0
    fig, axs = plt.subplots(2, 3)
    zoom = 2
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    for i in range(2):
        for j in range(3):
            real = data[names[counter + 1]]
            pred = data[names[counter]]
            y = list(range(len(real)))
            axs[i, j].scatter(real, y)
            axs[i, j].scatter(pred, y)
            n = names[counter].split('.')[0].split('-')
            n = n[:3]
            n = ' '.join(n)
            axs[i, j].set_title(n)
            counter += 2
    fig.savefig('Stack2-scatter.png')


def plot2():
    import os
    import re
    import pandas as pd
    import numpy as np

    table = 'Table4'
    stack = 'Stack2'

    tmp = ['ENN_MVO', 'ENN_PSO', 'LSTM_MVO', 'LSTM_PSO']
    names = list()
    for t in tmp:
        for i, j, k in os.walk('./Results/'):
            names2 = list()
            for l in k:
                if re.match(f'{table}_{stack}_(NO2|SO2)_{t}.xlsx', l):
                    names2.append(l)
            # print(names2)
            if len(names2) != 0:
                names.append(names2)

    dates = [1, 2, 7, 8, 9, 10]

    print(names)
    for name in names:
        new_data = list()
        data = pd.read_excel(f'./Results/{name[0]}')
        for i in dates:
            d = data.iloc[:, i]
            d = np.array(d)
            d = d.flatten()
            #****
            d = d[0]
            d = d.replace('\n', ',')
            d = eval(d)
            d = np.array(d)
            #****
            # if d.dtype == 'object':
            #     new_d = list()
            #     for ii in d:
            #         if type(ii) is str:
            #             new_d.append(ii[1:-1])
            #         else:
            #             new_d.append(ii)
            #     d = np.array(new_d, dtype='float64')
            #
            # print(d)
            new_data.append(d)

        data2 = pd.read_excel(f'./Results/{name[1]}')
        for i in dates:
            d = data2.iloc[:, i]
            d = np.array(d)
            d = d.flatten()
            # ****
            d = d[0]
            d = d.replace('\n', ',')
            d = eval(d)
            d = np.array(d)
            # ****
            # if d.dtype == 'object':
            #     new_d = list()
            #     for ii in d:
            #         if type(ii) is str:
            #             new_d.append(ii[1:-1])
            #         else:
            #             new_d.append(ii)
            #     d = np.array(new_d, dtype='float64')
            #
            # print(d)
            new_data.append(d)


        ddd = ['may', 'august', 'september', 'may', 'august', 'september']
        counter = 0
        fig, axs = plt.subplots(2, 3)
        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * zoom, h * zoom)
        for i in range(2):
            for j in range(3):
                real = new_data[counter]
                pred = new_data[counter + 1]
                axs[i, j].plot(real)
                axs[i, j].plot(pred)
                axs[i, j].set_title(ddd[int(counter / 2)])
                counter += 2
        filename = name[0].split('.')
        filename = filename[0].split('_')
        filename = filename[3] + '_' + filename[4]
        print(filename)
        fig.savefig(f'./draws/{table}_{stack}_{filename}.png')


def scatter2():
    import os
    import re
    import pandas as pd
    import numpy as np

    table = 'Table4'
    stack = 'Stack2'

    tmp = ['ENN_MVO', 'ENN_PSO', 'LSTM_MVO', 'LSTM_PSO']
    names = list()
    for t in tmp:
        for i, j, k in os.walk('./Results/'):
            names2 = list()
            for l in k:
                if re.match(f'{table}_{stack}_(NO2|SO2)_{t}.xlsx', l):
                    names2.append(l)
            # print(names2)
            if len(names2) != 0:
                names.append(names2)

    dates = [1, 2, 7, 8, 9, 10]

    print(names)
    for name in names:
        new_data = list()
        data = pd.read_excel(f'./Results/{name[0]}')
        for i in dates:
            d = data.iloc[:, i]
            d = np.array(d)
            d = d.flatten()
            #****
            # d = d[0]
            # d = d.replace('\n', ',')
            # d = eval(d)
            # d = np.array(d)
            #****
            if d.dtype == 'object':
                new_d = list()
                for ii in d:
                    if type(ii) is str:
                        new_d.append(ii[1:-1])
                    else:
                        new_d.append(ii)
                d = np.array(new_d, dtype='float64')
            #
            # print(d)
            new_data.append(d)

        data2 = pd.read_excel(f'./Results/{name[1]}')
        for i in dates:
            d = data2.iloc[:, i]
            d = np.array(d)
            d = d.flatten()
            # ****
            # d = d[0]
            # d = d.replace('\n', ',')
            # d = eval(d)
            # d = np.array(d)
            # ****
            if d.dtype == 'object':
                new_d = list()
                for ii in d:
                    if type(ii) is str:
                        new_d.append(ii[1:-1])
                    else:
                        new_d.append(ii)
                d = np.array(new_d, dtype='float64')
            #
            # print(d)
            new_data.append(d)


        ddd = ['may', 'august', 'september', 'may', 'august', 'september']
        counter = 0
        fig, axs = plt.subplots(2, 3)
        zoom = 2
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * zoom, h * zoom)
        for i in range(2):
            for j in range(3):
                real = new_data[counter]
                pred = new_data[counter + 1]
                y = list(range(len(real)))
                axs[i, j].scatter(real, y)
                axs[i, j].scatter(pred, y)
                axs[i, j].set_title(ddd[int(counter / 2)])
                counter += 2
        filename = name[0].split('.')
        filename = filename[0].split('_')
        filename = filename[3] + '_' + filename[4]
        print(filename)
        fig.savefig(f'./draws/{table}_{stack}_{filename}.png')

scatter2()