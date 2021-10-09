import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import constants as c
from classes import BigBody, probe

# Some Constants
N = int(1e3)
#tspan = np.linspace(0, 5*c.yr2s, N) # s
tspan = 5*c.yr2s

# Initial Condition
#Init = [c.rE*np.cos(np.pi/3), c.rE*np.sin(np.pi/3), 1e3, 1e3]
Init = [-1.4e11, 0, 0, 0, 0, 0]

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('#303030')


Sun   = BigBody(2e30, 0, 0, 0)
Earth = BigBody(6e24, c.rE, 0, 0)

SourceList = [Sun, Earth]
ProbeList = []
        

#Probe = probe(-1.4e11, 0, 0, -5e3)
Probe1 = probe(Init[0], Init[1], Init[2], Init[3], Init[4], Init[5], SourceList, tspan)
Probe2 = probe(Init[1], Init[0], Init[2], Init[4], Init[3], Init[5], SourceList, tspan)
ProbeList.append(Probe1)
ProbeList.append(Probe2)

# To record the position of the probe
X = np.zeros((len(ProbeList), int(tspan/c.dt)))
Y = np.zeros((len(ProbeList), int(tspan/c.dt)))
#aCor = []#np.zeros(int(tspan/c.dt))
#aCen = []#np.zeros(int(tspan/c.dt))
#aG   = []#np.zeros(int(tspan/c.dt))

for i in tqdm(range(int(tspan/c.dt))):
    for j, Probe in enumerate(ProbeList):
        Probe.update(c.dt)
        X[j,i] = Probe.x
        Y[j,i] = Probe.y
    #aCor.append(Probe.acor)
    #aCen.append(Probe.acen)
    #aG.append(Probe.ag)

sun = ax.scatter(0, 0, color='y', s=100)
earth = ax.scatter(c.rE, 0, color='deepskyblue', s=30, zorder=2)

def Plot_Lagrange_Point(fig=fig, ax=ax):
    R = c.rE
    ax.scatter(148.11e9, 0, color='w', s=10, zorder=3) # L1
    ax.scatter(151.1e9,  0, color='w', s=10, zorder=3) # L2
    ax.scatter(-149.6e9, 0, color='w', s=10, zorder=3) # L3
    ax.scatter(c.rE*np.cos(np.pi/3),  c.rE*np.sin(np.pi/3),  color='w', s=10, zorder=3) # L4
    ax.scatter(c.rE*np.cos(-np.pi/3), c.rE*np.sin(-np.pi/3), color='w', s=10, zorder=3) # L5
    return
Plot_Lagrange_Point()

# Earth Orbit
theta = np.linspace(0, 2*np.pi, 200, endpoint=True)
ax.plot(c.rE*np.cos(theta), c.rE*np.sin(theta), color='deepskyblue', linewidth=1, zorder=1)

# Gravi. Field Contour

#xx, yy = 


# Animations

#line, = ax.plot(X[0], Y[0], color='cyan', linestyle='-', linewidth=1)
dots  = ax.scatter([], [], color='cyan' , marker='o', s=5)


def update(i):
    #dots = ax.scatter(X[:,i], Y[:,i], color='cyan', marker='o', s=5)
    dots.set_offsets(np.stack(X[:,i], Y[:,i]))
    line = ax.plot(X[min(0, i-10):i], Y[min(0, i-10):i], color='cyan', linestyle='-', linewidth=1)
    #line.set_data(X[min(0, i-10):i], Y[min(0, i-10):i])
    #arracor = ax.arrow(X[i], Y[i], aCor[i][0]*c.resize, aCor[i][1]*resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='r', ec='r')
    #arracen = ax.arrow(X[i], Y[i], aCen[i][0]*c.resize, aCen[i][1]*resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='y', ec='y')
    #arrag   = ax.arrow(X[i], Y[i], aG[i][0]*c.resize,   aG[i][1]*resize,   head_width=c.arrowsize, head_length=c.arrowsize, fc='skyblue', ec='skyblue')
    return [dots, line]#, arracor, arracen, arrag]

#ax.plot(X, Y)
ax.set_xlim([-c.BoxSize, c.BoxSize])
ax.set_ylim([-c.BoxSize, c.BoxSize])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

ani = animation.FuncAnimation(
    fig=fig, 
    func=update, 
    frames=int(tspan/c.dt), 
    interval=1000/int(tspan/c.dt),
    blit=True, 
    repeat=False)

plt.show()
