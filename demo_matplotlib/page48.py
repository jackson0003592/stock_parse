from matplotlib import pyplot as plt, font_manager

plt.figure(figsize=(20, 8), dpi=80)

interval = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90]
width = [5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 30, 60]
quantity = [836, 2737, 3723, 3926, 3596, 1438, 3273, 642, 824, 613, 215, 47]

plt.bar(range(len(quantity)), quantity, width=1)

_x = [i - 0.5 for i in range(13)]
_xticks_labels = interval + [150]
plt.xticks(_x, _xticks_labels)
plt.grid(alpha=0.4)

plt.show()
