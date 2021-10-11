# This file stores the constants that we will need

import numpy as np
from classes import BigBody

# Here are constants
G = 6.67e-11
day2s = 86400 # s
yr2s = 86400*365.2425 # s
AU = 1.5e11 # m

Msun = 2e30

rE = AU # m
ME = 6e24 # kg
TE = yr2s # s

rJ = 5.204*AU
TJ = (rJ/rE)**(3/2)*yr2s
MJ = 317.8*ME

# Here are parameters
BoxSize = rJ*1.5 # Size of the animation box
dt = 0.5*day2s # Simulation time resolution
resize = 25*BoxSize
arrowsize = 2.5e-2*BoxSize
tspan = 10*yr2s # s
tail = 200 # frames

def Set_Sources(M1, M2, R):
    omega  = np.array([0, 0, np.sqrt(G*(M1+M2)/R**3)])
    Sun    = BigBody(M1, -R*(M2/(M1+M2)), 0, 0)
    Planet = BigBody(M2, R*(M1/(M1+M2)) , 0, 0)
    SourceList = [Sun, Planet]
    return SourceList, omega

SourceList, omega = Set_Sources(Msun, MJ, rJ)
