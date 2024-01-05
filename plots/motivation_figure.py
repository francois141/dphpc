import matplotlib.pyplot as plt
import numpy as np

# Create some fake data.


y = [10, 1,0.1,0.01,0.001]
x1 = [292, 80,49,52,47]
x2 = [1500, 136,11,1,0.2]

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(y, x1, '.-', label="Tiling")
ax1.plot(y, x2, '.-', label="No tiling")
ax1.set_xscale('log')
ax1.set_ylabel('V100')
ax1.set_xlabel('Density')
ax1.legend(loc="upper right")
ax1.invert_xaxis()
ax1.set_facecolor('#f5f5f5')
ax1.set_ylim([0, 1600])

ax2.plot(y, x1, '.-', label="Tiling")
ax2.plot(y, x2, '.-', label="No Tiling")
ax2.set_xscale('log')
ax2.set_ylabel('A100')
ax2.set_xlabel('Density')
ax2.legend(loc="upper right")
ax2.invert_xaxis()
ax2.set_facecolor('#f5f5f5')
ax2.set_ylim([0, 1600])

plt.savefig("output.png")
plt.show()