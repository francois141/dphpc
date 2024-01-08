import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

val = pd.read_csv("results/motivation/v100/results-v100.csv")
val = val[['competitor', 'dataset', 'comp_ns']]
val = val.groupby(by= ["competitor", "dataset"]).mean()
val['comp_ns'] /= 1000
print(val)

sns.set_theme(context="notebook", font_scale=1, style="darkgrid", rc={ "lines.linewidth": 2, "axes.linewidth": 1, "axes.edgecolor":"black", "xtick.bottom": True, "ytick.left": True })
sns.despine(left=True, bottom=False)

y = [0.25,0.1,0.01,0.001,0.0001,0.00001]

# A100
x1 = np.array([37959,13966,1175,152,42,40])
x2 = np.array([9301,4489,1389,1004,947,931])

# V100
v100_1 = np.array([63146,21573,935,120,44,42])
v100_2 = np.array([24466,10394,2634,1464,1461,1625])


#fig, (ax1, ax2) = plt.subplots(2, 1)
fig, ax1 = plt.subplots(figsize=(6,2.5))

ax1.plot(y, x2, '.-', label="Tiling")
ax1.plot(y, x1, '.-', label="No tiling")
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_ylabel('A100 runtime [ms]')
ax1.set_xlabel('Density (%)')
ax1.legend(loc="upper right")
ax1.invert_xaxis()
#ax1.set_ylim([0, 1600])

# ax2.plot(y, v100_2, '.-', label="Tiling")
# ax2.plot(y, v100_1, '.-', label="No Tiling")
# ax2.set_xscale('log')
# #ax2.set_yscale('log')
# ax2.set_ylabel('V100 runtime [ms]')
# ax2.set_xlabel('Density (%)')
# ax2.legend(loc="upper right")
# ax2.invert_xaxis()
#ax2.set_ylim([0, 1600])
plt.tight_layout()
plt.savefig("output.png", dpi = 300)
plt.show()