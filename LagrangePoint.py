import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation

# Some Constants
G = 6.67e-11
day2s = 86400 # s
yr2s = 86400*365.2425 # s
dt = day2s # s
tspan = 5*yr2s # s
omega = np.array([0, 0, 2*np.pi/yr2s]) # 1/s
rE = 149.6e9 # m

# Initial Condition
#Init = [rE*np.cos(np.pi/3), rE*np.sin(np.pi/3), 1e3, 1e3]
Init = [-1.4e11, 0, 0, 0]

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('#303030')

class BigBody:

    def __init__(self, M, x, y):
        self.M = M
        self.x = x
        self.y = y
        return

Sun   = BigBody(2e30, 0, 0)
Earth = BigBody(6e24, rE, 0)

SourceList = [Sun, Earth]
        
class probe:

    def __init__(self, x, y, vx, vy):
        self.x  = x
        self.y  = y
        self.vx = vx
        self.vy = vy
        return
    
    def get_a(self):
        
        # Creating an array to store force
        a = np.zeros(2)
        
        # Gravity
        for body in SourceList:
            vec_r = np.array([(self.x - body.x), (self.y - body.y)])
            r = np.linalg.norm(vec_r)
            a += -G*body.M/r**3*vec_r

        # Centrifugal
        acen = -np.cross(omega, np.cross(omega, np.array([self.x, self.y, 0])))[0:2]
        a += acen
        
        # Coriolis
        acor = -2*np.cross(omega, np.array([self.vx, self.vy, 0]))[0:2]
        a += acor
        
        # Euler force is 0 since dw/dt=0 (assuming circular motion of Earth)

        return a

    def update(self, dt):
        a = self.get_a()
        self.vx += a[0]*dt
        self.vy += a[1]*dt
        self.x += self.vx*dt
        self.y += self.vy*dt
        return

#Probe = probe(-1.4e11, 0, 0, -5e3)
Probe = probe(Init[0],Init[1],Init[2],Init[3],)

# To record the position of the probe
X = np.zeros(int(tspan/dt))
Y = np.zeros(int(tspan/dt))

for i in tqdm(range(int(tspan/dt))):
    X[i] = Probe.x
    Y[i] = Probe.y
    Probe.update(dt)

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

line, = ax.plot(X[0], Y[0], color='cyan', linestyle='-', linewidth=1)
dot,  = ax.plot([], [], color='cyan' , marker='o', markersize=5, markeredgecolor='black', linestyle='')

def update(i):
    dot.set_data(X[i], Y[i])
    line.set_data(X[:i], Y[:i])
    return [dot, line]

def init():
    dot.set_data(X[0], Y[0])
    line.set_data(X[0], Y[0])
    return [dot, line]


#ax.plot(X, Y)
ax.set_xlim([-2e11, 2e11])
ax.set_ylim([-2e11, 2e11])
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
