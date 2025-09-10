# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:58:47 2019

@author: qijin
"""
import numpy as np
import matplotlib.pyplot as plt
from hicosmo.MCMC import MCMC_class




x,y_obs,y_err=np.loadtxt('./data/sim_data.txt',unpack=True)

# plt.errorbar(x,y_obs,y_err,fmt='.',color='k',elinewidth=0.7,capsize=2,alpha=0.9,capthick=0.7)
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.xlim(0,3)

#%%

def y_th(a,b,c):
    return a*x**2+b*x+c

def chi2(theta):
    a,b,c=theta
    return np.sum((y_obs-y_th(a,b,c))**2/y_err**2)

params=[
        ['a',3.5,0,10],
        ['b',2,0,4],
        ['c',1,0,2],
        ]
#%%
MC=MCMC_class(params,chi2,'example',len(x))
MC.runMC()

#%%
from hicosmo.MCMC import MCplot

chains=[
        ['example',''],
        ]

pl=MCplot(chains)
# pl.plot3D(0)
# pl.plot2D([3,2])
# pl.plot1D(2)
pl.results

#%%
xx=np.arange(0,3,0.01)
plt.plot(xx,3.32*xx**2+1.28*xx+1.087)
plt.errorbar(x,y_obs,y_err,fmt='.',color='k',elinewidth=0.7,capsize=2,alpha=0.9,capthick=0.7)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(0,3)