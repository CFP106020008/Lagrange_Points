import constants as c
import numpy as np
from scipy.integrate import solve_ivp

class BigBody:
    # This class stores the body that is stationary in Earth co-rotating coord.
    # For now, they are Sun and Earth
    def __init__(self, M, x, y, z):
        self.M = M
        self.x = x
        self.y = y
        self.z = z
        return


class probe:

    def __init__(self, x, y, z, vx, vy, vz, SourceList, tspan, omega):
        self.x  = x
        self.y  = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.acor = np.array([0,0,0])
        self.acen = np.array([0,0,0])
        self.ag =   np.array([0,0,0])
        self.SourceList = SourceList
        self.tspan = tspan
        self.omega = omega
        return
    
    def get_a(self):
        
        # Creating an array to store force
        a = np.zeros(3)
        ag = np.zeros(3)
        # Gravity
        for body in self.SourceList:
            vec_r = np.array([(self.x - body.x), (self.y - body.y), (self.z - body.z)])
            r = np.linalg.norm(vec_r)
            ag += -c.G*body.M/r**3*vec_r
        self.ag = ag
        a += ag

        # Centrifugal
        acen = -np.cross(self.omega, np.cross(self.omega, np.array([self.x, self.y, self.z])))
        a += acen
        self.acen = acen
        
        # Coriolis
        acor = -2*np.cross(self.omega, np.array([self.vx, self.vy, self.vz]))
        a += acor
        self.acor = acor
        
        # Euler force is 0 since dw/dt=0 (assuming circular motion of Earth)

        return a
    
    def solve_path(self, tspan):
        sol = solve_ivp(function = self.get_a,
                        t_span = [0, self.tspan], 
                        y_0 = self.get_P(), 
                        t_eval = np.linspace(0, self.tspan, 1000),
                        method='DOP853')
        self.T  = sol.t
        Data    = sol.y
        self.X  = Data[0,:]
        self.Y  = Data[1,:]
        self.Z  = Data[2,:]
        self.VX = Data[3,:]
        self.VY = Data[4,:]
        self.VZ = Data[5,:]
        return
    
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
        k1 = self.get_dPdt()
        self.add_P(k1*dt/2)
        k2 = self.get_dPdt()
        self.assign_P(p0)
        self.add_P(k2*dt/2)
        k3 = self.get_dPdt()
        self.assign_P(p0)
        self.add_P(k3*dt)
        k4 = self.get_dPdt()
        self.assign_P(p0)
        self.add_P(dt*(k1+2*k2+2*k3+k4)/6)
        return




