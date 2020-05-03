# coding= utf-8
from matplotlib import pyplot as plt

x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

fig = plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y)

_xticks_lables = [i / 2 for i in range(4, 49)]
plt.xticks(_xticks_lables[::2])

# _yticks_lables =

plt.yticks(range(min(y), max(y) + 1))

plt.show()
