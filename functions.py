import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import constants as c
import classes as cl

def Plot_Static(SourceList, fig, ax):
    # Lagrange points
    Sun = SourceList[0]
    Planet = SourceList[1]
    R = Planet.x - Sun.x
    M = Planet.M + Sun.M
    ax.scatter(Sun.x, Sun.y, color='y', s=100, zorder=4)
    ax.scatter(Planet.x, Planet.y, color='deepskyblue', s=30, zorder=4)
    
    rpL1 = R*( (Planet.M/M) / ( 3 * (1-Planet.M/M) ) )**(1/3)
    ax.scatter(Planet.x - rpL1, 0, color='w', s=10, zorder=3) # L1
    ax.scatter(Planet.x + rpL1, 0, color='w', s=10, zorder=3) # L2
    
    RL3 = -R*(1+5/12*(Planet.M/M)/(1-Planet.M/M))
    ax.scatter(RL3, 0, color='w', s=10, zorder=3) # L3
    
    ax.scatter(R*np.cos(np.pi/3)+Sun.x,  R*np.sin(np.pi/3),  color='w', s=10, zorder=3) # L4
    ax.scatter(R*np.cos(-np.pi/3)+Sun.x, R*np.sin(-np.pi/3), color='w', s=10, zorder=3) # L5
    
    # Earth Orbit
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
        R = Planet.x - Sun.x
        mu = Planet.M / (Planet.M + Sun.M)
        def func(x):
            # check https://leancrew.com/all-this/2016/08/lagrange-points-redux/
            return 3*(1-mu)*x**3*(1-x+x**2/3)-mu*(1-x)**3*(1+x+x**2)
        sol = root(func, 1e-2)
        print('d', sol.x*R)
        xL1 = Planet.x - sol.x*R
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

def save_frame(i, ProbeList, fig, ax):
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
    set_plot_dimension(fig, ax)
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

def set_plot_dimensions(fig, ax):
    ax.set_xlim([-c.BoxSize, c.BoxSize])
    ax.set_ylim([-c.BoxSize, c.BoxSize])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return