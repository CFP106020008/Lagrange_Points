# This file stores the constants that we will need

import numpy as np
from classes import BigBody

# Here are constants
G = 6.6743e-11
day2s = 86400 # s
yr2s = 86400*365.2425 # s
AU = 1.5e11 # m

Msun = 2e30

# Earth
rE = AU # m
ME = 5.97219e24 # kg
TE = yr2s # s

# Jupiter
rJ = 5.204*AU
TJ = (rJ/rE)**(3/2)*yr2s
MJ = 317.8*ME

# Moon
rM = 384400e3 # m, radius of the moon's orbit
#rM = 389703e3 # m, radius of the moon's orbit
R = 1.215058560962404E-2
MM = ME*R/(1-R)

#MM = 7.3477e22 # kg
#TM = (4*np.pi**2/(G*(ME+MM))*rM**3)**0.5
TM = 382981
RM = 1737.1e3 # m, Radius of the moon
