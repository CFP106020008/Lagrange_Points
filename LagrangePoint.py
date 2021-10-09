import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import constants as c
from classes import BigBody, probe

# Some Constants
frames = int(c.tspan/c.dt)

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
Probe1 = probe(Init[0], Init[1], Init[2], Init[3], Init[4], Init[5], SourceList, c.tspan)
Probe2 = probe(Init[1], Init[0], Init[2], Init[4], Init[3], Init[5], SourceList, c.tspan)
ProbeList.append(Probe1)
ProbeList.append(Probe2)

# To record the position of the probe
X = np.zeros((len(ProbeList), int(frames)))
Y = np.zeros((len(ProbeList), int(frames)))
aCor = np.zeros((len(ProbeList), frames, 2))
aCen = np.zeros((len(ProbeList), frames, 2))
aG   = np.zeros((len(ProbeList), frames, 2))

for i in tqdm(range(int(frames))):
    for j, Probe in enumerate(ProbeList):
        Probe.update(c.dt)
        X[j,i] = Probe.x
        Y[j,i] = Probe.y
        aCor[j,i,:] = Probe.acor[:2]
        aCen[j,i,:] = Probe.acen[:2]
        aG[j,i,:] = Probe.ag[:2]


def Plot_Static(fig=fig, ax=ax):
    # Lagrange points
    sun = ax.scatter(0, 0, color='y', s=100)
    earth = ax.scatter(c.rE, 0, color='deepskyblue', s=30, zorder=2)
    ax.scatter(148.11e9, 0, color='w', s=10, zorder=3) # L1
    ax.scatter(151.1e9,  0, color='w', s=10, zorder=3) # L2
    ax.scatter(-149.6e9, 0, color='w', s=10, zorder=3) # L3
    ax.scatter(c.rE*np.cos(np.pi/3),  c.rE*np.sin(np.pi/3),  color='w', s=10, zorder=3) # L4
    ax.scatter(c.rE*np.cos(-np.pi/3), c.rE*np.sin(-np.pi/3), color='w', s=10, zorder=3) # L5
    # Earth Orbit
    theta = np.linspace(0, 2*np.pi, 200, endpoint=True)
    ax.plot(c.rE*np.cos(theta), c.rE*np.sin(theta), color='deepskyblue', linewidth=1, zorder=1)
    return
Plot_Static()


# Gravi. Field Contour

#xx, yy = 


# Animations

#line, = ax.plot(X[0], Y[0], color='cyan', linestyle='-', linewidth=1)
#dots  = ax.scatter([], [], color='cyan' , marker='o', s=5)


def update(i):
    lines = []
    dots = []
    arracors = []
    arracens = []
    arrags = []
    for j, Probe in enumerate(ProbeList):
        lines.append(ax.plot(X[j, min(0, i-10):i], Y[j, min(0, i-10):i], color='cyan', linestyle='-', linewidth=1)[0])
        dots.append( ax.plot(X[j,i], Y[j,i], color='cyan', linestyle='-', markersize=5, marker='o')[0])
        arracors.append(ax.arrow(X[j,i], Y[j,i], aCor[j,i,0]*c.resize, aCor[j,i,1]*c.resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='r', ec='r'))
        arracens.append(ax.arrow(X[j,i], Y[j,i], aCen[j,i,0]*c.resize, aCen[j,i,1]*c.resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='y', ec='y'))
        arrags.append(ax.arrow(X[j,i], Y[j,i], aG[j,i,0]*c.resize,   aG[j,i,1]*c.resize,   head_width=c.arrowsize, head_length=c.arrowsize, fc='skyblue', ec='skyblue')) 
    #dots = ax.scatter(X[:,i], Y[:,i], color='cyan', marker='o', s=5)
    #line.set_data(X[min(0, i-10):i], Y[min(0, i-10):i])
    #arracor = ax.arrow(X[i], Y[i], aCor[i][0]*c.resize, aCor[i][1]*resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='r', ec='r')
    #arracen = ax.arrow(X[i], Y[i], aCen[i][0]*c.resize, aCen[i][1]*resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='y', ec='y')
    #arrag   = ax.arrow(X[i], Y[i], aG[i][0]*c.resize,   aG[i][1]*resize,   head_width=c.arrowsize, head_length=c.arrowsize, fc='skyblue', ec='skyblue')
    return dots + lines + arracors + arracens + arrags

#ax.plot(X, Y)
ax.set_xlim([-c.BoxSize, c.BoxSize])
ax.set_ylim([-c.BoxSize, c.BoxSize])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

ani = animation.FuncAnimation(
    fig=fig, 
    func=update, 
    frames=int(frames), 
    interval=1000/int(frames),
    blit=True, 
    repeat=False)

plt.show()
