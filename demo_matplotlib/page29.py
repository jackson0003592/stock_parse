from matplotlib import pyplot as plt, font_manager

a = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]
b = [1, 0, 3, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
x = range(11, 31)

my_font = font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, a, label='自己', color='cyan', linestyle=":")
plt.plot(x, b, label='同桌', color='orange', linestyle="--")

plt.xticks(x, ["{}岁".format(i) for i in x], fontproperties=my_font)
plt.yticks(range(9))

plt.grid(alpha=0.4, linestyle=":")
plt.legend(prop=my_font, loc='upper left')

plt.show()
