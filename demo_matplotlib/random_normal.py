import random
import matplotlib.pyplot as plt

walk = []
for _ in range(1000):
    walk.append(random.normalvariate(0,1))

plt.hist(walk, bins=30)
plt.show()