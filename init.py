# This file is used to generate 
# initial conditions for the simulation

import numpy as np
import matplotlib.pyplot as plt
#from LagrangePoint import ProbeList
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

def DRO(Dx, Dy, n, stdx, stdy, tspan, SourceList):
    '''
    Dx and Dy are the distance "relative" to the secondary source.
    '''
    ProbeList = []
    Main = SourceList[0]
    Secondary = SourceList[1]
    omega = np.array([0, 0, np.sqrt(c.G*(Main.M+Secondary.M)/(Secondary.x**2+Secondary.y**2)**(3/2))])
    r = (Dx**2 + Dy**2)**0.5
    rr = ((Secondary.x+Dx)**2 + (Secondary.x+Dy)**2)**0.5
    vc = (c.G*Secondary.M/r)**0.5
    x = np.random.normal(loc=Secondary.x+Dx, scale=stdx, size=n)
    print(x)
    y = np.random.normal(loc=Secondary.y+Dy, scale=stdy, size=n)
    print(y)
    vx = np.ones(n)*vc*Dy/r
    #vy = -np.ones(n)*vc*Dx/r
    vy = -np.ones(n)*510
    for i in range(n):
        ProbeList.append(cl.probe(x[i], y[i], 0, vx[i], vy[i], 0, SourceList, tspan, omega))
    return ProbeList
