import os
import json
import matplotlib.pyplot as plt


items = []
for i, j, k in os.walk('./Results 3/'):
    for i in k:
        item = list()
        h = i.split(' ')
        if h[-1] == 'list':
            with open(f'./Results 3/{i}', 'r') as f:
                data = f.read()
                data = data.split('\n')
                item = list()
                item.append((h[0], h[1], h[2], h[3], h[4]))
                for d in data:
                    if d != '':
                        d = json.loads(d)
                        m = min(d)
                        item.append(m)
                items.append(item)

print(items[0])
print(len(items[0]))

for i in items:
    print(i[0])
    plt.plot(i[1:])
    plt.savefig(f'./figs/{i[0][0]} {i[0][1]} {i[0][2]} {i[0][3]} {i[0][4]}.png')
    plt.clf()
