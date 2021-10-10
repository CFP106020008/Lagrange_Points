import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import constants as c
from classes import BigBody, probe
from matplotlib.colors import SymLogNorm

# Some Constants
frames = int(c.tspan/c.dt)

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
fig.set_facecolor('#303030')
ax.set_facecolor('#303030')


SourceList = c.SourceList
ProbeList = []
'''
for i in range(5):
    for j in range(5):
        ProbeList.append(probe(c.BoxSize/5*i, c.BoxSize/5*j, 0, 0, 0, 0, SourceList, c.tspan))
'''
#Init = [c.BoxSize/5*2, c.BoxSize/5*0, 0, 0, 0, 0, SourceList, c.tspan]
Init1 = [c.rE*np.cos(np.pi/3)+SourceList[0].x-5e4, c.rE*np.sin(np.pi/3), 0, 0, 0, 0, SourceList, c.tspan]
Init2 = [c.rE*np.cos(np.pi/3)+SourceList[0].x-5e5, c.rE*np.sin(np.pi/3), 0, 0, 0, 0, SourceList, c.tspan]
Init3 = [c.rE*np.cos(np.pi/3)+SourceList[0].x-5e6, c.rE*np.sin(np.pi/3), 0, 0, 0, 0, SourceList, c.tspan]
        

Probe1 = probe(Init1[0], Init1[1], Init1[2], Init1[3], Init1[4], Init1[5], SourceList, c.tspan)
Probe2 = probe(Init2[0], Init2[1], Init2[2], Init2[3], Init2[4], Init2[5], SourceList, c.tspan)
Probe3 = probe(Init3[0], Init3[1], Init3[2], Init3[3], Init3[4], Init3[5], SourceList, c.tspan)
ProbeList.append(Probe1)
ProbeList.append(Probe2)
ProbeList.append(Probe3)

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


def Plot_Static(SourceList, fig=fig, ax=ax):
    # Lagrange points
    Sun = SourceList[0]
    Planet = SourceList[1]
    R = Planet.x - Sun.x
    ax.scatter(Sun.x, Sun.y, color='y', s=100, zorder=4)
    ax.scatter(Planet.x, Planet.y, color='deepskyblue', s=30, zorder=4)
    #ax.scatter(148.11e9, 0, color='w', s=10, zorder=3) # L1
    #ax.scatter(151.1e9,  0, color='w', s=10, zorder=3) # L2
    #ax.scatter(-149.6e9, 0, color='w', s=10, zorder=3) # L3
    #ax.scatter(-Rp,  0,  color='w', s=10, zorder=3) # L3
    ax.scatter(R*np.cos(np.pi/3)+Sun.x,  R*np.sin(np.pi/3),  color='w', s=10, zorder=3) # L4
    ax.scatter(R*np.cos(-np.pi/3)+Sun.x, R*np.sin(-np.pi/3), color='w', s=10, zorder=3) # L5
    # Earth Orbit
    #theta = np.linspace(0, 2*np.pi, 200, endpoint=True)
    #ax.plot(Planet.x*np.cos(theta), Planet.x*np.sin(theta), color='blue', linewidth=1, zorder=3)
    return
Plot_Static(SourceList)


# Gravi. Field Contour
L = np.linspace(-c.BoxSize, c.BoxSize, 1000)
xx, yy = np.meshgrid(L, L)
g = np.zeros((1000, 1000))
for Source in SourceList: # Gravity from all sources
    #print(Source.M, Source.x, Source.y)
    #print(np.shape(xx))
    r = ((xx - Source.x)**2 + (yy - Source.y)**2)**0.5
    g += -c.G*Source.M/r
R = ((xx-0)**2 + (yy-0)**2)**0.5 # Distance from the origin
g -= 0.5*(R*np.linalg.norm(c.omega))**2 # Centrifugal Force
Lv=np.linspace(0, np.max(g), 70)
#ax.contourf(xx, yy, g, 
#            zorder=1, 
#            levels = Lv,
#            #norm=SymLogNorm(linthresh=1, base=10),
#            extend='both',
#            alpha = 0.5,
#            cmap='gray')
ax.contour( xx, yy, g, 
            zorder=2, 
            levels=np.linspace(np.max(g)-0.3*np.std(g), np.max(g), 20),
            #norm=SymLogNorm(linthresh=1, base=10),
            colors='#606060', 
            linewidths=1,
            linestyles='-')


# Animations
def update(i):
    lines = []
    dots = []
    arracors = []
    arracens = []
    arrags = []
    for j, Probe in enumerate(ProbeList):
        lines.append(ax.plot(X[j, max(0, i-c.tail):i], Y[j, max(0, i-c.tail):i], color='cyan', linestyle='-', linewidth=1)[0])
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
    frames=np.arange(0, frames, 2), 
    interval=1000/int(frames),
    blit=True, 
    repeat=False)

def save_frame(i, fig=fig, ax=ax):
    ax.cla()
    for j, Probe in enumerate(ProbeList):
        ax.plot(X[j, max(0, i-c.tail):i], Y[j, max(0, i-c.tail):i], color='cyan', linestyle='-', linewidth=1)
        ax.plot(X[j,i], Y[j,i], color='cyan', linestyle='-', markersize=5, marker='o')
        ax.arrow(X[j,i], Y[j,i], aCor[j,i,0]*c.resize, aCor[j,i,1]*c.resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='r', ec='r')
        ax.arrow(X[j,i], Y[j,i], aCen[j,i,0]*c.resize, aCen[j,i,1]*c.resize, head_width=c.arrowsize, head_length=c.arrowsize, fc='y', ec='y')
        ax.arrow(X[j,i], Y[j,i], aG[j,i,0]*c.resize,   aG[j,i,1]*c.resize,   head_width=c.arrowsize, head_length=c.arrowsize, fc='skyblue', ec='skyblue') 
    ax.set_xlim([-c.BoxSize, c.BoxSize])
    ax.set_ylim([-c.BoxSize, c.BoxSize])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    Lv=np.linspace(np.max(g)-np.std(g)*0.7, np.max(g), 50)
    ax.contourf(xx, yy, g, 
                zorder=1, 
                levels = Lv,
                norm=SymLogNorm(linthresh=1, base=10),
                extend='both',
                alpha = 0.5,
                #cmap='cividis')
                cmap='gray')
    Plot_Static(SourceList)
    fig.savefig("./Frames/frame_{:04d}.jpg".format(i), dpi=300)
'''
for i in tqdm(range(frames)):
    save_frame(i)
'''
#ani.save("movie.mp4", dpi=300, fps=20)

plt.show()
