# This file stores the constants that we will need

import numpy as np

# Here are constants
G = 6.67e-11
day2s = 86400 # s
yr2s = 86400*365.2425 # s
omega = np.array([0, 0, 2*np.pi/yr2s]) # 1/s
rE = 149.6e9 # m

# Here are parameters
BoxSize = 2e11
dt = day2s
resize = 5e12
arrowsize = 5e9
tspan = 5*yr2s
N = int(1e3)
