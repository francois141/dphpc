import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv = pd.read_csv("dummy_measures.csv")
arr = np.array(csv).transpose()

fig, ax = plt.subplots()

keys = np.unique(arr[0])

for key in keys:
       ax.plot(arr[1][arr[0] == key], arr[2][arr[0] == key], label=key)

ax.set(xlabel='Size', ylabel='Speed',
       title='Benchmark SDDMM - dphpc')
ax.grid()

plt.legend(loc="upper left")
fig.savefig("benchmark.png")
#plt.show()