import json
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.stats as stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.axes_grid import AxesGrid
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

s = json.load( open("./bmh_matplotlibrc.json") )
matplotlib.rcParams.update(s)


bkgcolor = "#F8F8F8"
fillcolor = [.5,.75,.95]
linecolor = [.2,.35,.60]

fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
m = 0
sd = 2
x = np.arange(-7,7,0.1)
y = stats.norm.pdf(x,m, sd)
ax.plot(x,y, color=linecolor, lw=1.5)
ax.fill_between(x, 0, y, color=fillcolor, alpha=0.4)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$f(x)$', fontsize=18)
plt.ylim(0,np.max(y)*1.2)
ax.grid(b=True, which='both', color='0.35')
plt.show()


bkgcolor = "#F8F8F8"
fillcolor = [.5,.75,.95]
linecolor = [.2,.35,.60]
fillcolor1 = [.95,.5,.75]
linecolor1 = [.60,.2,.35]

fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
m = 0
sd = 1
m1=3.5
sd1=.1
x = np.arange(-5,5,0.1)
y = stats.norm.pdf(x,m, sd)
y1 = stats.norm.pdf(x,m1, sd1)
ax.plot(x,5*y, color=linecolor, lw=1.5)
ax.fill_between(x, 0, 5*y, color=fillcolor, alpha=0.4)
ax.plot(x,y1, color=linecolor, lw=1.5)
ax.fill_between(x, 0, y1, color=fillcolor, alpha=0.4)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$p(x)$', fontsize=18)
plt.ylim(0,np.max(y1)*1.2)
ax.grid(b=True, which='both', color='0.35')
plt.show()




fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
m = 0
sd = 2
x = np.arange(-7,7,0.1)
y = stats.norm.pdf(x,m, sd)
ax.plot(x,y, color=linecolor, lw=1.5)
ax.fill_between(x, 0, y, color=fillcolor, alpha=0.4)
ax.fill_between(x, 0, y, where=x<=m-2*sd, facecolor=[.9,.1,.1])
ax.fill_between(x, 0, y, where=x>=m+2*sd, facecolor=[.9,.1,.1])
ax.annotate('$2.5\%$', xy=(m+2.2*sd, stats.norm.pdf(m+2.2*sd,m, sd)),  xycoords='data',
            xytext=(m+3*sd, 0.05), textcoords='data',
           arrowprops=dict(arrowstyle="->", #linestyle="dashed",
                                color=linecolor,
                                linewidth=1,
                                shrinkA=15, shrinkB=5,
                                patchA=None,
                                patchB=None,
                                connectionstyle="arc3, rad=0.",
                                ),
            horizontalalignment='right', verticalalignment='top'
            )
ax.annotate('$2.5\%$', xy=(m-2.2*sd, stats.norm.pdf(m-2.2*sd,m, sd)),  xycoords='data',
            xytext=(m-3*sd, 0.05), textcoords='data', 
           arrowprops=dict(arrowstyle="->", #linestyle="dashed",
                                color=linecolor,
                                linewidth=1,
                                shrinkA=15, shrinkB=5,
                                patchA=None,
                                patchB=None,
                                connectionstyle="arc3, rad=0.",
                                ),
            horizontalalignment='right', verticalalignment='top'
            )
#plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.xticks([m-3*sd, m-2*sd,m-sd, m, m+sd,m+2*sd, m+3*sd], 
  ['$\mu-3\sigma$', '$\mu-2\sigma$','$\mu-\sigma$', '$\mu$', '$\mu+\sigma$', '$\mu+2\sigma$', '$\mu+3\sigma$'])
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$f(x)$', fontsize=18)
plt.ylim(0,np.max(y)*1.2)
ax.grid(b=True, which='both', color='0.35')
plt.savefig('./normalsigma.pdf')
plt.show()


fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
l = 1.7
x = np.arange(1,17)
y = stats.poisson.pmf(x,l)
ax.bar(x,y, color=fillcolor)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$p(x)$', fontsize=18)
ax.grid(b=True, which='both', color='0.35')
plt.show()



    
rv = stats.multivariate_normal([0, 0], [[2.0, 3.5], [3.5, 15]])

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X,Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z = rv.pdf(pos)

#Z = flux_qubit_potential(X, Y).T

fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
#im = ax.imshow(Z, cmap=cm.rainbow, extent=[-1, 1, -1, 1])
im = ax.imshow(Z, origin='lower', extent=(min(x), max(x), min(y), max(y)), alpha=.8)
im.set_interpolation('bilinear')
#cb = fig.colorbar(im, ax=ax)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax.grid(b=True, which='both', color='0.35')
plt.show()


rv = stats.multivariate_normal([0, 0], [[10.0, 0.0], [0.0, 10.0]])

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X,Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z = rv.pdf(pos)

#Z = flux_qubit_potential(X, Y).T

fig = plt.figure(figsize=(5,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
#im = ax.imshow(Z, cmap=cm.rainbow, extent=[-1, 1, -1, 1])
im = ax.imshow(Z, origin='lower', extent=(min(x), max(x), min(y), max(y)), alpha=.8)
im.set_interpolation('bilinear')
#cb = fig.colorbar(im, ax=ax)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
ax.set_xlabel('$w_0$', fontsize=18)
ax.set_ylabel('$w_1$', fontsize=18)
ax.grid(b=True, which='both', color='0.35')
plt.show()



fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, axisbg=bkgcolor)
ax.contour(X, Y, Z, 20, alpha=.75, cmap='jet', vmin=abs(Z).min(), vmax=abs(Z).max(), extent=(min(x), max(x), min(y), max(y)))
#cnt = ax.contour(Z, cmap=cm.jet, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax.grid(b=True, which='both', color='0.35')
plt.axis('equal')
plt.show()


fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, projection='3d', axisbg=bkgcolor)
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True)
#cb = fig.colorbar(p, shrink=0.5)
ax.view_init(20, 45)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax.set_zlabel('$f(x,y)$', fontsize=18)
plt.show()

fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, projection='3d', axisbg=bkgcolor)
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, alpha=0.25)
ax.view_init(60, 30)
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax.set_zlabel('$f(x,y)$', fontsize=18)
plt.show()


fig = plt.figure(figsize=(11,5))
fig.patch.set_facecolor('white')
ax= fig.add_subplot(1,1,1, projection='3d', axisbg=bkgcolor)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.9, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#cset = ax.contour(X, Y, Z, zdir='z', offset=1, cmap=cm.jet)
#ax.contour(X, Y, Z, 18, zdir='x', offset=2, cmap=cm.jet)
#ax.contour(X, Y, Z, zdir='y', offset=2, cmap=cm.jet)
#ax.set_xlim3d(-np.pi, 2*np.pi);
#ax.set_ylim3d(0, 3*np.pi);
#ax.set_zlim3d(-np.pi, 2*np.pi);
plt.setp( ax.get_xticklabels(), visible=False)
plt.setp( ax.get_yticklabels(), visible=False)
plt.setp( ax.get_zticklabels(), visible=False)
ax.set_xlabel('$x$', fontsize=18)
ax.set_ylabel('$y$', fontsize=18)
ax.set_zlabel('$f(x,y)$', fontsize=18)
plt.show()




import math
import operator

def dirichlet_pdf(x, alpha):
  return (math.gamma(sum(alpha)) / 
          reduce(operator.mul, [math.gamma(a) for a in alpha]) *
          reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))
          
alpha=[0.1,0.1]
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X,Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
Z = dirichlet_pdf(X,alpha)