from matplotlib import pyplot as plt, font_manager
import random
import matplotlib

# font = {
#     'family': 'MicroSoft YaHei',
#     'weight': 'bold'
# }
#
# matplotlib.rc('font', **font)

my_font = font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc', size='small')


x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

plt.figure(figsize=(30, 8))

plt.plot(x, y)

_x = list(x)
_xticks_lables = ["10点{}分".format(i) for i in range(60)]
_xticks_lables += ["11点{}分".format(i) for i in range(60)]
plt.xticks(_x[::3], _xticks_lables[::3], rotation=40, fontproperties=my_font)

plt.xlabel("时间", fontproperties=my_font)
plt.ylabel("温度 单位(°C)", fontproperties=my_font)
plt.title("10点到12点每分钟温度变化情况", fontproperties=my_font)

plt.show()
