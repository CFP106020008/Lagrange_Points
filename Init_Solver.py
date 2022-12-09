import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import constants as c
import functions as f
import init
from classes import BigBody, probe
from matplotlib.colors import SymLogNorm

# Important parameters
#BoxSize = c.rM*1.5 # Size of the animation box
BoxSize = 100000e3 # m
dt = 5e-3*c.day2s # Simulation time resolution
resize = 25*BoxSize
arrowsize = 2.5e-2*BoxSize
tspan = 0.5*c.yr2s # s
tail = 2000 # frames
SAVE_VIDEO = False

# Some Constants
frames = int(tspan/dt)
Nframe = 600

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
fig.set_facecolor('#303030')
ax.set_facecolor('#303030')

#SourceList, omega = Set_Sources(Msun, 0.05*Msun, rE)
SourceList, omega = f.Set_Sources(c.ME, c.MM, c.rM)
#ProbeList = init.Random_Around(rE*np.cos(np.pi/3)+SourceList[0].x, rE*np.sin(np.pi/3), 20, 1e7, 1e7)

def Get_x(r, v):
    ProbeList = init.DRO(r, 0, 1, 0, 0, tspan, SourceList, reverse=False, vy=v)
    time = 0
    for i in tqdm(range(int(frames))):
        time += dt
        for j, Probe in enumerate(ProbeList):
            Probe.update(dt)
        if ((i > 100) and (abs(Probe.y) < 100e3)) and (Probe.x > c.rM):
            break
    return Probe.x - (SourceList[1].x + r)

def Bisection(r, v0, v1, torx):
    vnewx = 1e99
    while abs(vnewx) > torx:
        vnew = (v0 + v1)/2
        v0x = Get_x(r, v0)
        v1x = Get_x(r, v1)
        vnewx = Get_x(r, vnew)
        if v0x*vnewx < 0:
            v1 = vnew
        else: 
            v0 = vnew
    return vnew

def Secant(r, torx):
    v0 = 500
    v1 = 520
    v0x = Get_x(r, v0)
    v1x = Get_x(r, v1)
    while abs(v1x) > torx:
        vs = v1 - v1x/((v1x - v0x)/(v1 - v0))
        v0 = v1
        v1 = vs
        #err = v1x - (c.rM + r)
        v0x = v1x
        v1x = Get_x(r, vs)
    return v1
            
#print(Bisection(50000e3, 500, 520, 1000e3))

#vs = np.linspace(500,520,50)
#r = 50000e3*np.ones(len(vs))
#
#solx = list(map(Get_x, r, vs))
#solv = vs[np.argmin(solx)]
#plt.plot(solx)
#plt.show()




