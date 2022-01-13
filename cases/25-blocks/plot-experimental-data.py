import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

header = []

with open("25-blocks.start") as fp:
    for i, line in enumerate(fp):
        if i > 4: break
        header.append(line)

cast = lambda ans, i : int(ans) if i < 2 else float(ans)

ncols, nrows, xmin, ymin, cellsize = [ cast(line.split()[1], i) for i, line in enumerate(header) ]

y_range = 0.2 - ymin

row = int(y_range / cellsize)

# reading from the top down
# +6 to skip the 6 header lines
row_read = nrows - row + 6

x_range_start = 4 - xmin
x_range_end   = 8 - xmin

col_start = int(x_range_start / cellsize)
col_end   = int(x_range_end   / cellsize)

cols = range(col_start,col_end)

sim_data = np.loadtxt( fname="./results/results-5.wd", skiprows=row_read, max_rows=1, usecols=cols )

plt.plot([i * cellsize + xmin for i in cols], sim_data)
plt.show()