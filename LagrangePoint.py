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

omega = c.omega

SourceList = c.SourceList
ProbeList = []
        

#Probe = probe(-1.4e11, 0, 0, -5e3)
Probe1 = probe(Init[0], Init[1], Init[2], Init[3], Init[4], Init[5], SourceList, c.tspan)
#Probe2 = probe(Init[1], Init[0], Init[2], Init[4], Init[3], Init[5], SourceList, c.tspan)
ProbeList.append(Probe1)
#ProbeList.append(Probe2)

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


def Plot_Static(Rp, fig=fig, ax=ax):
    # Lagrange points
    sun = ax.scatter(0, 0, color='y', s=100, zorder=3)
    planet = ax.scatter(Rp, 0, color='deepskyblue', s=30, zorder=3)
    #ax.scatter(148.11e9, 0, color='w', s=10, zorder=3) # L1
    #ax.scatter(151.1e9,  0, color='w', s=10, zorder=3) # L2
    #ax.scatter(-149.6e9, 0, color='w', s=10, zorder=3) # L3
    ax.scatter(-Rp,  0,  color='w', s=10, zorder=3) # L3
    ax.scatter(Rp*np.cos(np.pi/3),  Rp*np.sin(np.pi/3),  color='w', s=10, zorder=3) # L4
    ax.scatter(Rp*np.cos(-np.pi/3), Rp*np.sin(-np.pi/3), color='w', s=10, zorder=3) # L5
    # Earth Orbit
    theta = np.linspace(0, 2*np.pi, 200, endpoint=True)
    ax.plot(Rp*np.cos(theta), Rp*np.sin(theta), color='deepskyblue', linewidth=1, zorder=1)
    return
Plot_Static(c.rE)


# Gravi. Field Contour
L = np.linspace(-c.BoxSize, c.BoxSize, 1000)
xx, yy = np.meshgrid(L, L)
g = np.zeros((1000, 1000))
for Source in SourceList: # Gravity from all sources
    print(Source.M, Source.x, Source.y)
    print(np.shape(xx))
    r = ((xx - Source.x)**2 + (yy - Source.y)**2)**0.5
    g += -c.G*Source.M/r
R = ((xx-0)**2 + (yy-0)**2)**0.5 # Distance from the origin
g -= 0.5*(R*np.linalg.norm(c.omega))**2 # Centrifugal Force
Contour = ax.contourf(xx, yy, g, 
                     zorder=1, 
                     levels=np.linspace(np.max(g)-np.std(g),
                                        np.max(g)/2,
                                        20), 
                     #linewidths=1,
                     cmap='gray')

# Animations
def update(i):
    lines = []
    dots = []
    arracors = []
    arracens = []
    arrags = []
    for j, Probe in enumerate(ProbeList):
        lines.append(ax.plot(X[j, max(0, i-10):i], Y[j, max(0, i-10):i], color='cyan', linestyle='-', linewidth=1)[0])
        dots.append( ax.plot(X[j,i], Y[j,i], color='cyan', linestyle='-', markersize=5, marker='o')[0])
        arracors.append(ax.arrow(X[j,i], Y[j,i], aCor[j,i,0]*c.resize, aCor[j,i,1]*c.resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='r', ec='r'))
        arracens.append(ax.arrow(X[j,i], Y[j,i], aCen[j,i,0]*c.resize, aCen[j,i,1]*c.resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='y', ec='y'))
        arrags.append(ax.arrow(X[j,i], Y[j,i], aG[j,i,0]*c.resize,   aG[j,i,1]*c.resize,   head_width=c.arrowsize, head_length=c.arrowsize, fc='skyblue', ec='skyblue')) 
    return dots + lines + arracors + arracens + arrags

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
