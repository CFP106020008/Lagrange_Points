import numpy as np
import matplotlib.pyplot as plt
import constants as c
from classes import BigBody, probe

Sun    = BigBody(2e30, 0,    0, 0)
Planet = BigBody(c.MJ, c.rJ, 0, 0)

SourceList = [Sun, Planet]
omega = np.sqrt((Sun.M + Planet.M)*c.G/c.rJ**3)

n = int(1e3)
L = np.linspace(-c.BoxSize, c.BoxSize, n)
xx, yy = np.meshgrid(L, L)
g = np.zeros((n, n))
plane = np.ones((n,n))
print(xx)
for Source in SourceList:
    print(Source.M, Source.x, Source.y)
    print(np.shape(xx))
    r = ((xx - plane*Source.x)**2 + (yy - plane*Source.y)**2)**0.5
    g += -c.G*Source.M/r
R = (xx**2 + yy**2)**0.5
#print(g)
g += -0.5*(R*omega)**2
plt.contourf(xx, yy, g, zorder=5, levels=np.linspace(-2.7e7, -2.55e7, 50))
plt.colorbar()
plt.show()




