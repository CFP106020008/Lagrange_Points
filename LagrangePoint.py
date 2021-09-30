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
Init = [-1.4e11, 0, 0, 0, 0, 0]

# Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('#303030')

class BigBody:

    def __init__(self, M, x, y, z):
        self.M = M
        self.x = x
        self.y = y
        self.z = z
        return

Sun   = BigBody(2e30, 0, 0, 0)
Earth = BigBody(6e24, rE, 0, 0)

SourceList = [Sun, Earth]
        
class probe:

    def __init__(self, x, y, z, vx, vy, vz):
        self.x  = x
        self.y  = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.acor = np.array([0,0,0])
        self.acen = np.array([0,0,0])
        self.ag =   np.array([0,0,0])
        return
    
    def get_a(self):
        
        # Creating an array to store force
        a = np.zeros(3)
        ag = np.zeros(3)
        # Gravity
        for body in SourceList:
            vec_r = np.array([(self.x - body.x), (self.y - body.y), (self.z - body.z)])
            r = np.linalg.norm(vec_r)
            ag += -G*body.M/r**3*vec_r
        self.ag = ag
        a += ag

        # Centrifugal
        acen = -np.cross(omega, np.cross(omega, np.array([self.x, self.y, self.z])))
        a += acen
        self.acen = acen
        
        # Coriolis
        acor = -2*np.cross(omega, np.array([self.vx, self.vy, self.vz]))
        a += acor
        self.acor = acor
        
        # Euler force is 0 since dw/dt=0 (assuming circular motion of Earth)

        return a

    def get_P(self): # The position of the probe in phase space (since there are so many "p")
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz])

    def add_P(self, k):
        self.x  += k[0]
        self.y  += k[1]  
        self.z  += k[2]  
        self.vx += k[3]
        self.vy += k[4]
        self.vz += k[5]

    def assign_P(self, P):
        self.x  = P[0]
        self.y  = P[1]  
        self.z  = P[2]  
        self.vx = P[3]
        self.vy = P[4]
        self.vz = P[5]

    def get_dPdt(self):
        a = self.get_a()
        return np.array([self.vx, self.vy, self.vz, a[0], a[1], a[2]])

    def update(self, dt):
        p0 = self.get_P()
        k1 = self.get_dPdt()*dt
        self.add_P(k1)
        k2 = self.get_dPdt()*dt
        self.assign_P(p0)
        self.add_P((k1+k2)/2)
        return

#Probe = probe(-1.4e11, 0, 0, -5e3)
Probe = probe(Init[0],Init[1],Init[2],Init[3], Init[4], Init[5])

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
