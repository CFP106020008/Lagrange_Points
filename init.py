# This file is used to generate 
# initial conditions for the simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import constants as c
import classes as cl


def Random_Around(x, y, n, stdx, stdy):
    ProbeList = []
    x = np.random.normal(loc=x, scale=stdx, size=n)
    y = np.random.normal(loc=y, scale=stdy, size=n)
    for i in range(n):
        ProbeList.append(cl.probe(x[i], y[i], 0, 0, 0, 0, c.SourceList, c.tspan))
    return ProbeList


