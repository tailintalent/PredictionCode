# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:36:41 2017
create a rotation matrix for Eigenworms to get better separation of velocity and turns.
With unmodified output, sinusoidal modulation remain in turn component.
@author: monika
"""
# standard modules
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
# custom modules
import dataHandler as dh
import makePlots as mp
from sklearn import linear_model

###############################
# transform by linear fit in 2d followed by rotation. 
###############################

def findRotationMatrix(xS, yS, zS):
    """calculate rotation matrix to have corrected eigenworm projections."""
    dist = zS**2+xS**2
    #make transformation of z-y plane using angle from linear fit
    xS = np.reshape(xS, (-1,1))
    zS = np.reshape(zS, (-1,1))
    turns = np.argpartition(dist, -250)[:-250]
#    #use ransac to obtain a fit of all non-turn points 
#    #(use the fact there are more non-turns than turn points
#    reg = linear_model.RANSACRegressor()
#    reg.fit(zS, xS)
#    #use ransac outliers to fit
#    # turns only - fit to align to z-axis
#    turns = np.where(reg.inlier_mask_==False)[0]
    # fit outliers (turns) only
    reg = linear_model.LinearRegression()
    reg.fit(xS[turns], zS[turns]) 
    # outliers need to align with z
    theta = np.pi/2.-np.arctan(reg.coef_[0])[0]
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[0,1,0],[c,0, -s], [s,0, c]])

    return R

# load data

dataPars = {'medianWindow':1, # smooth eigenworms with median filter of that size, must be odd
            'savGolayWindow':13, # savitzky-golay window for angle velocity derivative. must be odd
            'rotate':False, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5 # savitzky-golay window for red and green channel
            }

typ='AML70'

# GCamp6s; lite-1
if typ =='AML70': 
    folder = "AML70_moving/{}_MS/"
    dataLog = "AML70_moving/AML70_datasets.txt"
    outLoc = "AML70_moving/Analysis/"
# GCamp6s 
if typ =='AML32': 
    folder = "AML32_moving/{}_MS/"
    dataLog = "AML32_moving/AML32_datasets.txt"
    outLoc = "AML32_moving/Analysis/"
##### GFP
elif typ =='AML18': 
    folder = "AML18_moving/{}_MS/"
    dataLog = "AML18_moving/AML18_datasets.txt"
    outLoc = "AML18_moving/Analysis/"
# output is stored here
elif typ =='AML32imm': 
    folder = "AML32_immobilized/{}_MS/"
    dataLog = "AML32_immobilized/AML32_immobilized_datasets.txt"#loc = 'AML18_moving/'
#folder = os.path.join(loc,"{}_MS/")
#dataLog = os.path.join(loc,"AML18_datasets.txt")
## output is stored here
outfile = os.path.join(outLoc,"../Rotationmatrix.dat")
dataSets = dh.loadMultipleDatasets(dataLog,folder, dataPars)
nWorms = len(dataSets)
overwrite = True#False # if True fits new rotation matrix and overwrites old one!
###############################
# concatenate Eigenworms from all datasets -- create new rotation matrix
###############################
if overwrite:
    xS, yS, zS = [],[],[]
    for kindex, key in enumerate(dataSets.keys()):
        data = dataSets[key]
        xtmp, ytmp, ztmp = data['Behavior']['Eigenworm1'], data['Behavior']['Eigenworm2'], data['Behavior']['Eigenworm3']
        xS.append(xtmp)
        yS.append(ytmp)
        zS.append(ztmp)
        
    xS = np.concatenate(xS)
    yS = np.concatenate(yS)
    zS = np.concatenate(zS)
    
    R = findRotationMatrix(xS, yS, zS)
    hdr = """Common rotation for Eigenworms. In python:\,
    xN,yN, zN = np.array(np.dot(R, np.vstack([xS,yS,zS]))), 
    where R is the 3x3 rotation matrix, and xS, yS, zS are the wholebrain pipeline Eigenworms. """
    
    np.savetxt(outfile, R, header = hdr )
else:
    R = np.loadtxt(outfile)

print 'Done loading data'


# show projections for each dataset
fig = plt.figure('Eigenworm projections',(6.8, nWorms*3.4))
gs = gridspec.GridSpec(nWorms,1, hspace=1, wspace=1, top=0.95, bottom = 0.25, left = 0.15, right =0.9)

for kindex, key in enumerate(dataSets.keys()):
    data = dataSets[key]
    xS, yS, zS = data['Behavior']['Eigenworm1'], data['Behavior']['Eigenworm2'], data['Behavior']['Eigenworm3']
        
    gsobj = gs[kindex]        
    #make transformation of z-y plane using angle from linear fit
    xN,yN, zN = np.array(np.dot(R, np.vstack([xS,yS,zS])))
    plt.title(key)
    axes = mp. plot2DProjections(xN,yN, zN, fig, gsobj)
    # plot old point clouds
    s, a = 0.05, 0.25
    axes[0].scatter(xS+10,yS,color='k', s=s, alpha = a, zorder=5)
    axes[1].scatter(xS+10,zS,color='k', s=s, alpha = a)
    axes[2].scatter(zS+10,yS,color='k', s=s, alpha = a)
    for ax in axes:
        plt.sca(ax)
        plt.xlim([-20,20])
        plt.ylim([-20,20])

#plt.tight_layout()
gs.tight_layout(fig, rect=[0, 0.05, 1, 0.95], h_pad = 0.9)
plt.show()

theta2 = np.unwrap(np.arctan2(yS, xS))
velo2 = dh.savitzky_golay(theta2, window_size=17, order=7, deriv=1, rate=1)
theta = np.unwrap(np.arctan2(yN, xN))
velo = dh.savitzky_golay(theta, window_size=41, order=3, deriv=1, rate=1)
turn = dh.savitzky_golay(zN, window_size=9, order=5)
# make sure that the angle velocity positive = fwd
plt.figure('Rotated Eigenworms')
plt.subplot(211)
plt.plot(velo, label = 'phase velocity after rotation')
v = dataSets[key]['Behavior']['CMSVelocity']
plt.plot(v*np.max(velo)/np.max(v), label = 'CMS velocity (rescaled)')
plt.ylabel('New phase velocity')
plt.legend()
plt.subplot(212)
plt.plot(zN, label='new turns')
plt.plot(zS, label='old turns')
plt.ylabel('New turns')
plt.legend()
plt.show()
