import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.animation as animation
import constants as c
import functions as f
import init
from classes import BigBody, probe
from Init_Solver import Bisection, Secant

# Important parameters
#BoxSize = c.rM*1.5 # Size of the animation box
BoxSize = 700000e3 # m
dt = 5e-3*c.day2s # Simulation time resolution
resize = 25*BoxSize
arrowsize = 2.5e-2*BoxSize
tspan = 0.5*c.yr2s # s
tail = 50000 # frames
SAVE_VIDEO = True

# Some Constants
frames = int(tspan/dt)
Nframe = 1200

#ProbeList = init.Random_Around(rE*np.cos(np.pi/3)+SourceList[0].x, rE*np.sin(np.pi/3), 20, 1e7, 1e7)
ProbeList = []

LU = 389703e3
TU = 382981
SourceList, omega = f.Set_Sources(c.ME, c.MM, LU)
#omega = np.array([0,0,1/TU])
#ProbeList.append(probe(  4.8784941344943100E-1*LU, 5.5201264559703600E-1*LU, 0, 
#                        -4.5897050958509600E-1*LU/TU, 2.1938634831283776E-1*LU/TU, 0, 
#                        SourceList, tspan, omega))
ProbeList += init.Read_JPL_ics( './setups/axial_L4.csv', LU, TU,
                                np.linspace(10,1300,3, endpoint=True).astype(int), SourceList, tspan, omega)
ProbeList += init.Read_JPL_ics( './setups/axial_L5.csv', LU, TU,
                                np.linspace(10,1300,3, endpoint=True).astype(int), SourceList, tspan, omega)
#ProbeList += init.Read_JPL_ics( './setups/2-1_Resonant.csv', LU, TU,
#                                np.linspace(0,1067,5, endpoint=True).astype(int), SourceList, tspan, omega)
#ProbeList += init.Read_JPL_ics( './setups/DRO.csv', LU, TU,
#                                np.linspace(300,900,5, endpoint=True).astype(int), SourceList, tspan, omega)


#ProbeList += init.DRO(20000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(20000e3, 500e3))
#ProbeList += init.DRO(40000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(30000e3, 500e3))
#ProbeList += init.DRO(60000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(40000e3, 500e3))
#ProbeList += init.DRO(80000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(50000e3, 500e3))
#ProbeList += init.DRO(60000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(60000e3, 500e3))
#ProbeList += init.DRO(70000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(70000e3, 500e3))
#ProbeList += init.DRO(80000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=Secant(80000e3, 500e3))
#ProbeList += init.DRO(60000e3, 0, 1, 0, 0, tspan, SourceList, reverse=True, vy=Secant(60000e3, 500e3))
#ProbeList += init.DRO(70000e3, 0, 1, 0, 0, tspan, SourceList, reverse=True, vy=Secant(70000e3, 500e3))
#ProbeList += init.DRO(80000e3, 0, 1, 0, 0, tspan, SourceList, reverse=True, vy=Secant(80000e3, 500e3))
#ProbeList += init.DRO(60000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False)
#ProbeList += init.DRO(50000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False)
#ProbeList += init.DRO(40000e3, 0, 1, 0, 0, tspan, SourceList, reverse=False)
#ProbeList += init.DRO(60000e3, 0, 1, 0, 0, tspan, SourceList, reverse=True)
#ProbeList += init.DRO(50000e3, 0, 1, 0, 0, tspan, SourceList, reverse=True)
#ProbeList += init.DRO(40000e3, 0, 1, 0, 0, tspan, SourceList, reverse=True)

#ProbeList += init.DRO(c.RM + 200e3, 0, 1, 0, 0, tspan, SourceList, reverse=False)
