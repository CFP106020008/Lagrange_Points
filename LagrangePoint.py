import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import constants as c
import functions as f
import init
from classes import BigBody, probe
from matplotlib.colors import SymLogNorm
from Init_Solver import Bisection, Secant

# Import the initial setup
from setups.Earth_Moon_System import *

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
fig.set_facecolor('#303030')
ax.set_facecolor('#303030')

#SourceList, omega = Set_Sources(Msun, 0.05*Msun, rE)

# To record the position of the probes
T = np.zeros((len(ProbeList), frames))
X = np.zeros((len(ProbeList), frames))
Y = np.zeros((len(ProbeList), frames))
aCor = np.zeros((len(ProbeList), frames, 2))
aCen = np.zeros((len(ProbeList), frames, 2))
aG   = np.zeros((len(ProbeList), frames, 2))

for i in tqdm(range(int(frames))):
    for j, Probe in enumerate(ProbeList):
        Probe.update(dt)
        T[j,i] = dt*i
        X[j,i] = Probe.x
        Y[j,i] = Probe.y
        aCor[j,i,:] = Probe.acor[:2]
        aCen[j,i,:] = Probe.acen[:2]
        aG[j,i,:] = Probe.ag[:2]

# Lagrange point positions
f.Plot_Static(SourceList, fig, ax)

# Gravi. Field Contour
f.Plot_Contour(SourceList, fig, ax, BoxSize)

# Animations
def update(i):
    lines = []
    dots = []
    arracors = []
    arracens = []
    arrags = []
    for j, Probe in enumerate(ProbeList):
        lines.append(ax.plot(X[j, max(0, i-tail):i], Y[j, max(0, i-tail):i], color='cyan', linestyle='-', linewidth=1)[0])
        dots.append( ax.plot(X[j,i], Y[j,i], color='cyan', linestyle='-', markersize=5, marker='o')[0])
        arracors.append(ax.arrow(X[j,i], Y[j,i], aCor[j,i,0]*resize, aCor[j,i,1]*resize, head_width=arrowsize, head_length=arrowsize, fc='r', ec='r'))
        arracens.append(ax.arrow(X[j,i], Y[j,i], aCen[j,i,0]*resize, aCen[j,i,1]*resize, head_width=arrowsize, head_length=arrowsize, fc='y', ec='y'))
        arrags.append(ax.arrow(X[j,i], Y[j,i], aG[j,i,0]*resize,   aG[j,i,1]*resize,   head_width=arrowsize, head_length=arrowsize, fc='skyblue', ec='skyblue')) 
    return dots + lines + arracors + arracens + arrags

# Set Boxsize, etc.
f.set_plot_dimensions(fig, ax, BoxSize, center=[SourceList[0].x, SourceList[0].y])

ani = animation.FuncAnimation(
    fig=fig, 
    func=update, 
    frames=np.arange(0, frames, 10), 
    interval=1000/int(frames),
    blit=True, 
    repeat=False)
#ani.save("movie.mp4", dpi=300, fps=20)

if SAVE_VIDEO:
    for i, frame in enumerate(tqdm(np.linspace(0, frames-1, Nframe).astype(int))):
        f.save_frame(   i, dt, frame, SourceList, ProbeList, 
                        X, Y, aCor, aCen, aG, 
                        fig, ax, BoxSize, tail, resize, arrowsize, 
                        center=[SourceList[0].x, SourceList[0].y])
else:
    plt.show()
