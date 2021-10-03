import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import constants as c

# Some Constants
N = int(1e3)
tspan = np.linspace(0, 5*c.yr2s, N) # s

# Initial Condition
#Init = [rE*np.cos(np.pi/3), rE*np.sin(np.pi/3), 1e3, 1e3]
Init = [-1.4e11, 0, 0, 0, 0, 0]

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('#303030')


Sun   = BigBody(2e30, 0, 0, 0)
Earth = BigBody(6e24, rE, 0, 0)

SourceList = [Sun, Earth]
        

#Probe = probe(-1.4e11, 0, 0, -5e3)
Probe = probe(Init[0],Init[1],Init[2],Init[3], Init[4], Init[5], SourceList, tspan)

# To record the position of the probe
X = np.zeros(int(tspan/dt))
Y = np.zeros(int(tspan/dt))
aCor = []#np.zeros(int(tspan/dt))
aCen = []#np.zeros(int(tspan/dt))
aG   = []#np.zeros(int(tspan/dt))

for i in tqdm(range(int(tspan/dt))):
    X[i] = Probe.x
    Y[i] = Probe.y
    aCor.append(Probe.acor)
    aCen.append(Probe.acen)
    aG.append(Probe.ag)
    Probe.update(dt)

#print(aCor)

sun = ax.scatter(0, 0, color='y', s=100)
earth = ax.scatter(rE, 0, color='deepskyblue', s=30, zorder=2)

def Plot_Lagrange_Point(fig=fig, ax=ax):
    R = rE
    ax.scatter(148.11e9, 0, color='w', s=10, zorder=3) # L1
    ax.scatter(151.1e9,  0, color='w', s=10, zorder=3) # L2
    ax.scatter(-149.6e9, 0, color='w', s=10, zorder=3) # L3
    ax.scatter(rE*np.cos(np.pi/3),  rE*np.sin(np.pi/3),  color='w', s=10, zorder=3) # L4
    ax.scatter(rE*np.cos(-np.pi/3), rE*np.sin(-np.pi/3), color='w', s=10, zorder=3) # L5
    return
Plot_Lagrange_Point()

# Earth Orbit
theta = np.linspace(0, 2*np.pi, 200, endpoint=True)
ax.plot(rE*np.cos(theta), rE*np.sin(theta), color='deepskyblue', linewidth=1, zorder=1)

# Gravi. Field Contour

X = np.linspace()
#xx, yy = 


# Animations

line, = ax.plot(X[0], Y[0], color='cyan', linestyle='-', linewidth=1)
dot,  = ax.plot([], [], color='cyan' , marker='o', markersize=5, markeredgecolor='black', linestyle='')

resize = 5e12
arrowsize = 5e9

def update(i):
    dot.set_data(X[i], Y[i])
    line.set_data(X[:i], Y[:i])
    arracor = ax.arrow(X[i], Y[i], aCor[i][0]*resize, aCor[i][1]*resize, head_width=arrowsize, head_length=arrowsize, fc='r', ec='r')
    arracen = ax.arrow(X[i], Y[i], aCen[i][0]*resize, aCen[i][1]*resize, head_width=arrowsize, head_length=arrowsize, fc='y', ec='y')
    arrag   = ax.arrow(X[i], Y[i], aG[i][0]*resize,   aG[i][1]*resize,   head_width=arrowsize, head_length=arrowsize, fc='skyblue', ec='skyblue')
    return [dot, line, arracor, arracen, arrag]

def init():
    dot.set_data(X[0], Y[0])
    line.set_data(X[0], Y[0])
    return [dot, line]


#ax.plot(X, Y)
ax.set_xlim([-c.Boxsize, c.Boxsize)
ax.set_ylim([-c.Boxsize, c.Boxsize)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

ani = animation.FuncAnimation(
    fig=fig, 
    func=update, 
    frames=len(X), 
    init_func=init, 
    interval=1000/len(X), 
    blit=True, 
    repeat=False)

plt.show()
