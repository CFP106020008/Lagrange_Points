# This file stores the constants that we will need

import numpy as np
from classes import BigBody

# Here are constants
G = 6.67e-11
day2s = 86400 # s
yr2s = 86400*365.2425 # s
AU = 1.5e11 # m

Msun = 2e30

# Earth
rE = AU # m
ME = 6e24 # kg
TE = yr2s # s

# Jupiter
rJ = 5.204*AU
TJ = (rJ/rE)**(3/2)*yr2s
MJ = 317.8*ME

# Moon
rM = 384400e3 # m
MM = 7.3477e22 # kg
TM = (4*np.pi**2/(G*(ME+MM))*rM**3)**0.5
