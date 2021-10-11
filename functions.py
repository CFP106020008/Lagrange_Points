import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import constants as c
import classes as cl

# Physics
def Find_Lagrange_Points(SourceList):
    # check https://leancrew.com/all-this/2016/08/lagrange-points-redux/
    Sun = SourceList[0]
    Planet = SourceList[1]
    R = Planet.x - Sun.x
    mu = Planet.M / (Planet.M + Sun.M)
    def func_L1(x):
        return 3*(1-mu)*x**3*(1-x+x**2/3)-mu*(1-x)**3*(1+x+x**2)
    sol = root(func_L1, 1e-2)
    L1 = [Planet.x - sol.x*R, 0]
    def func_L2(x):
        #return -3*(1-mu)*x**3*(1+x+x**2/3)+mu*(1-x**3)*(1+x)**2
        return (1-mu)*( (1-(1+x)**3) / (1+x)**2 ) + mu*((1-x**3)/x**2)
    sol = root(func_L2, 1e-2)
    L2 = [Planet.x + sol.x*R, 0]
    L3 = [-(1+5/12*mu/(1-mu))*R, 0]
    L4 = [R*np.cos( np.pi/3)+Sun.x, R*np.sin(np.pi/3)]
    L5 = [R*np.cos(-np.pi/3)+Sun.x, R*np.sin(-np.pi/3)]
    return [L1, L2, L3, L4, L5]

# Visualization
def Plot_Static(SourceList, fig, ax):
    # Lagrange points
    Sun = SourceList[0]
    Planet = SourceList[1]
    ax.scatter(Sun.x, Sun.y, color='y', s=100, zorder=4)
    ax.scatter(Planet.x, Planet.y, color='deepskyblue', s=30, zorder=4)
    
    LagrangePoints = Find_Lagrange_Points(SourceList)
    ax.scatter(LagrangePoints[0][0], 0, color='w', s=10, zorder=3) # L1
    ax.scatter(LagrangePoints[1][0], 0, color='w', s=10, zorder=3) # L2
    ax.scatter(LagrangePoints[2][0], 0, color='w', s=10, zorder=3) # L3
    ax.scatter(LagrangePoints[3][0], LagrangePoints[3][1],  color='w', s=10, zorder=3) # L4
    ax.scatter(LagrangePoints[4][0], LagrangePoints[4][1], color='w', s=10, zorder=3) # L5
    
    # Planet Orbit
    #theta = np.linspace(0, 2*np.pi, 200, endpoint=True)
    #ax.plot(Planet.x*np.cos(theta), Planet.x*np.sin(theta), color='blue', linewidth=1, zorder=3)
    return

def Plot_Contour(SourceList, fig, ax, fill=False):
    L = np.linspace(-c.BoxSize, c.BoxSize, 1000)
    xx, yy = np.meshgrid(L, L)
    g = np.zeros((1000, 1000))
    for Source in SourceList: # Gravity from all sources
        r = ((xx - Source.x)**2 + (yy - Source.y)**2)**0.5
        g += -c.G*Source.M/r
    R = ((xx-0)**2 + (yy-0)**2)**0.5 # Distance from the origin
    g -= 0.5*(R*np.linalg.norm(c.omega))**2 # Centrifugal Force
    def U_L1(SourceList):
        Sun = SourceList[0]
        Planet = SourceList[1]
        xL1 = Find_Lagrange_Points(SourceList)[0][0]
        U = -c.G*Sun.M/abs(xL1 - Sun.x) - c.G*Planet.M/abs(Planet.x - xL1) - 0.5*(c.omega[2]*xL1)**2
        return U
    Lv=np.linspace(0, np.max(g), 70)
    print('maxg', np.max(g))
    print('UL1',U_L1(SourceList))
    if fill:
        ax.contourf(xx, yy, g, 
                    zorder=1, 
                    levels = Lv,
                    #norm=SymLogNorm(linthresh=1, base=10),
                    extend='both',
                    alpha = 0.5,
                    cmap='gray')
    ax.contour( xx, yy, g, 
                zorder=2, 
                #levels=np.linspace(np.max(g)-0.3*np.std(g), np.max(g), 20),
                levels=np.linspace(U_L1(SourceList)[0], np.max(g), 20),
                #norm=SymLogNorm(linthresh=1, base=10),
                colors='#606060', 
                linewidths=1,
                linestyles='-')
    return

def set_plot_dimensions(fig, ax):
    ax.set_xlim([-c.BoxSize, c.BoxSize])
    ax.set_ylim([-c.BoxSize, c.BoxSize])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return

def save_frame(frame, i,
               SourceList, ProbeList, 
               X, Y, aCor, aCen, aG,
               fig, ax):
    ax.cla()
    for j, Probe in enumerate(ProbeList):
        ax.plot(X[j, max(0, i-c.tail):i], Y[j, max(0, i-c.tail):i], color='cyan', linestyle='-', linewidth=1)
        ax.plot(X[j,i], Y[j,i], color='cyan', linestyle='-', markersize=5, marker='o')
        ax.arrow(X[j,i], Y[j,i], aCor[j,i,0]*c.resize, aCor[j,i,1]*c.resize, zorder = 3, head_width=c.arrowsize, head_length=c.arrowsize, fc='r', ec='r')
        ax.arrow(X[j,i], Y[j,i], aCen[j,i,0]*c.resize, aCen[j,i,1]*c.resize, zorder = 3, head_width=c.arrowsize, head_length=c.arrowsize, fc='y', ec='y')
        ax.arrow(X[j,i], Y[j,i], aG[j,i,0]*c.resize,   aG[j,i,1]*c.resize,   zorder = 3, head_width=c.arrowsize, head_length=c.arrowsize, fc='skyblue', ec='skyblue') 
    ax.set_xlim([-c.BoxSize, c.BoxSize])
    ax.set_ylim([-c.BoxSize, c.BoxSize])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    set_plot_dimensions(fig, ax)
    Plot_Static(SourceList, fig, ax)
    Plot_Contour(SourceList, fig, ax, fill=False)
    fig.savefig("./Frames/frame_{:04d}.jpg".format(frame), dpi=300, facecolor='#303030')

