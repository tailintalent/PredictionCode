
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 3 - Sparse linear model predicts behavior
@author: monika
"""
import numpy as np
import matplotlib as mpl
import os
#
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.ticker as mtick
from  sklearn.metrics.pairwise import pairwise_distances
#from matplotlib_venn import venn2
 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
# custom pip
#import svgutils as svg
#
import makePlots as mp
import dataHandler as dh
import dimReduction as dr

from stylesheet import *
from pycpd import deformable_registration, rigid_registration


# suddenly this isn't imported from stylesheet anymore...
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["font.size"] = 12
fs = mpl.rcParams["font.size"]
################################################
#
#function for aligning two sets of data
#
################################################
def registerWorms(Xref, X, dim=3):
    """use pycpd code to register two worms"""
    # first we rigid
    registration = rigid_registration
    reg = registration(Xref[:,:dim], X[:,:dim], tolerance=1e-10)
    reg.register(callback=None)
    # ...then non rigid
    registration =deformable_registration
    reg = registration(Xref[:,:dim],reg.TY, tolerance=1e-5)
    reg.register(callback=None)
    return reg.TY
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML70', 'AML175']:
    for condition in ['moving', 'chip']:# ['moving', 'immobilized', 'chip']:
        folder = '{}_{}/'.format(typ, condition)
        dataLog = '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
        outLoc = "Analysis/{}_{}_results.hdf5".format(typ, condition)
        outLocData = "Analysis/{}_{}.hdf5".format(typ, condition)
        
        try:
            # load multiple datasets
            dataSets = dh.loadDictFromHDF(outLocData)
            keyList = np.sort(dataSets.keys())
            results = dh.loadDictFromHDF(outLoc) 
            # store in dictionary by typ and condition
            key = '{}_{}'.format(typ, condition)
            data[key] = {}
            data[key]['dsets'] = keyList
            data[key]['input'] = dataSets
            data[key]['analysis'] = results
        except IOError:
            print typ, condition , 'not found.'
            pass
print 'Done reading data.'

################################################
#
# create figure 1: This is twice the normal size
#
################################################
# we will select a 'special' dataset here, which will have all the individual plots
# select a special dataset - moving AML32. Should be the same as in fig 2
movingAML32 ='BrainScanner20180709_100433'
moving = data['AML32_moving']['input'][movingAML32]
movingAnalysis = data['AML32_moving']['analysis'][movingAML32]
# negative bends are ventral for this worm
fig = plt.figure('Fig - 3 : Neuron locations and overlap', figsize=(4.5, 2.5*2.25))
#fig.patch.set_alpha(0.0)
# this gridspec makes one example plot of a heatmap with its PCA
gs1 = gridspec.GridSpec(3, 2, width_ratios = [2,1], height_ratios = [0.75,1,1])
gs1.update(left=0.08, right=0.99, wspace=0.45, bottom = 0.01, top=0.98, hspace=0.2)

#motionNeurons = ['AVA', 'ASI', 'AIY','AIB','AIA','AIZ','AVD', 'RIM', 'SMB', 'RMD', 'SMD', 'RIV']

# from kato et al suppl. table
motionNeurons =  ['AIA', 'AIB', 'AIY', 'AIZ', 'AVA', 'AVB', 'AVD', 'DVA', 'RIA', 'RIB', 'RIM', 'RIV', 'RMD', 'RME', 'SIB', 'SMB', 'SMD']
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B']
y0 = 0.99
locations = [(0,y0),   (0.58,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
letters = ['C', 'D']
y0 = 0.66
locations = [(0,0.66),  (0,0.3), (0.76,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='top',)
#letters = ['F', 'G', 'H']
#y0 = 0.45
#locations = [(0,y0),  (0.32,y0), (0.73,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row - number of neurons, overlap and crossprediction
#
################################################
# weights
flag = 'ElasticNet'
avWeights = movingAnalysis[flag]['AngleVelocity']['weights']
avRelevant = np.where(np.abs(avWeights)>0)[0]
tWeights = movingAnalysis[flag]['Eigenworm3']['weights']
tRelevant = np.where(np.abs(tWeights)>0)[0]
notRelevant = (np.where(np.abs(avWeights==0)*np.abs(tWeights==0)))[0]

# one example
time = moving['Neurons']['TimeFull']
label = 'AngleVelocity'
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']

axNN = plt.subplot(gs1[0,0])

# number of neurons
for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, 0), (B2, 1.5)], [axNN, axNN ]):

    alldata = []
    # color and offset
    c = colors[0]
    xoff = colors[1]
    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            results=  dset[idn][flag][behavior]
            noNeurons = len(results['weights'][np.abs(results['weights'])>0])
            Ntotal = len(results['weights'])
            print noNeurons, noNeurons*1.0/Ntotal*100
            keep.append(noNeurons*1.0/Ntotal*100)
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        axR2.scatter(np.zeros(len(keep))+rnd1+xoff, keep, marker = marker, c = c, edgecolor=c, alpha=0.5)
        alldata.append(keep)
    alldata = np.reshape(np.concatenate(alldata), (-1,1))
    
    mkStyledBoxplot(axR2, [-0.25+xoff, 0.25+xoff], alldata.T, [c], ['GCamp6s'], scatter=False, dx=1)
    #print '% neurons (mean, stdev, N, s.e.m)', np.mean(alldata), np.std(alldata), len(alldata), np.std(alldata)/np.sqrt(len(alldata))
    #print 'No neurons (mean, stdev, N)', np.mean(alldata[1]), np.std(alldata[1]), len(alldata)
axNN.set_xlim([-1, 2.0])
axNN.set_xticks([-0.5,-0.5+xoff])
axNN.set_xticklabels(['Velocity', 'Body \n curvature'], rotation = 0)
axNN.set_ylabel('# neurons')
axNN.set_ylim([0,100])
moveAxes(axNN, 'right', 0.05)

########show overlap
axOverlap = plt.subplot(gs1[0,1])
venn= []
for key, marker in zip(['AML32_moving', 'AML70_chip'], ['o', "^"]):
        dset = data[key]['analysis']
        keep = []
        for idn in dset.keys():
            
            results=  dset[idn][flag]
            avWeights = results['AngleVelocity']['weights']
            avRelevant = np.where(np.abs(avWeights)>0)[0]
            tWeights = results['Eigenworm3']['weights']
            tRelevant = np.where(np.abs(tWeights)>0)[0]
            notRelevant = (np.where(np.abs(avWeights==0)*np.abs(tWeights==0)))[0]
            
            N = 1.0*len(avRelevant)+1.0*len(tRelevant)
            # this is what appears at least once
            unique = len(np.unique(np.concatenate([avRelevant, tRelevant])))
            
            overlap = len(np.intersect1d(avRelevant, tRelevant))
            
            zero = len(notRelevant)
            
            venn.append([unique/N, overlap/N, zero/N])
            keep.append([unique/N, overlap/N, zero/N])
        keep = np.array(keep)
        rnd1 = np.random.rand(len(keep))*0.2
        axOverlap.scatter(np.zeros(len(keep))+rnd1, keep[:,1], marker = marker, c =L1, edgecolor=None, alpha=0.5)
        #axOverlap.scatter(np.zeros(len(keep))+rnd1+1, keep[:,1], marker = marker, c = L2, edgecolor=None, alpha=0.5)
       
# plot boxplot of overlap
venn = np.array(venn)
print venn.shape
#mkStyledBoxplot(axOverlap, [-0.25,0.75,], venn.T, [L1, L2], ['unique','overlap'], scatter = False)
mkStyledBoxplot(axOverlap, [-0.1,0.75,], venn.T[1:2], [L1], ['unique'], scatter = False,dx=0.5, rotate=False)
axOverlap.set_xlim([-0.5, 0.25])
axOverlap.set_ylim([-0.0, 1.1])
axOverlap.set_ylabel('Fraction', labelpad=0)


######## converse prediction

#axNN = plt.subplot(gs1[0,2])
#flag = 'ConversePredictionEN'
## prediction with opposing set of neurons
#for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, 0), (B2, 2.25)], [axNN, axNN ]):
#
#    alldata = []
#    # color and offset
#    c = colors[0]
#    xoff = colors[1]
#    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
#        dset = data[key]['analysis']
#        keep = []
#        for idn in dset.keys():
#            results=  dset[idn]['ElasticNet'][behavior]
#            
#            results2=  dset[idn][flag][behavior]
#            keep.append([results['scorepredicted'], results2['scorepredicted']])
#            
#        keep = np.array(keep)
#        
#        rnd1 = np.random.rand(len(keep))*0.2
#        axR2.scatter(np.zeros(len(keep[:,0]))+rnd1+xoff+0.05, keep[:,0], marker = marker, c = c, edgecolor=c, alpha=0.5)
#        axR2.scatter(np.zeros(len(keep[:,1]))+rnd1+xoff+1, keep[:,1], marker = marker, c = c, edgecolor=c, alpha=0.5)
#        alldata.append(keep)
#    alldata = np.array(np.concatenate(alldata, axis=0))
#    mkStyledBoxplot(axR2, [-0.25+xoff, 0.75+xoff], alldata.T, [c, c], ['GCamp6s', 'GCamp6s'], scatter=False, dx=1)
#axNN.axhline(color='k', linestyle=':')
#axNN.set_xlim([-0.5, 3.75])
#axNN.set_xticks([-0.25,0.75, -0.25+xoff,  0.75+xoff])
#axNN.set_xticklabels(['Velocity (Full set)', 'Turn neurons', 'Turn (Full set)', 'Velocity neurons'])
#axNN.set_ylabel('R$^2 (Testset)$')
#axNN.set_ylim([-0.5,0.75])

################################################
#
#second row
#
################################################

# plot projections of neurons
s0,s1,s2 = 25,25,25 # size of gray, red, blue neurons
dim = False
flag = 'ElasticNet'
markers = ['p', '^', '*', 'X', '+', '8', 's', '3', '>']
weightLocs = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs1[1:,:], wspace=0.0, hspace=0)
#axweight = plt.subplot(weightLocs[0,0], aspect='equal', clip_on=False)
#axweight.set_title('Velocity neurons')
#axweight2 = plt.subplot(weightLocs[0,1], aspect='equal', clip_on=False)
#axweight2.set_title('Turn neurons')
axweight3 = plt.subplot(weightLocs[0,0], aspect='equal', clip_on=False)
#axweight3.set_title('Velocity neurons projected on Atlas', fontsize=fs)
axweight4 = plt.subplot(weightLocs[1, 0], aspect='equal', clip_on=False)
#axweight4.set_title('Body curvature \n neurons projected on Atlas', fontsize=fs)

# find the dataset with most neurons
#Xref = np.zeros((1,1))
#for key in ['AML32_moving', 'AML70_chip']:
#    for idn in data[key]['input'].keys():
#        X = np.copy(data[key]['input'][idn]['Neurons']['Positions'].T)
#        if len(X[0])>len(Xref[0]):
#           Xref = X
           
# use the moving dataset as reference
Xref = np.copy(moving['Neurons']['Positions'].T)
Xref -=np.mean(Xref,axis=0)
# load atlas data
neuron2D = 'utility/celegans277positionsKaiser.csv'
labels = np.loadtxt(neuron2D, delimiter=',', usecols=(0), dtype=str)
neuronAtlas2D = np.loadtxt(neuron2D, delimiter=',', usecols=(1,2))
relevantIds = (neuronAtlas2D[:,0]>-0.1)#*(neuronAtlas2D[:,0]<0.15)
A = neuronAtlas2D[relevantIds]
A[:,0] = -A[:,0]
labels = labels[relevantIds]
A -=np.mean(A, axis=0)
A /= np.ptp(A, axis=0)
A*= 1.2*np.ptp(Xref[:,:2], axis=0)

# plot all data
AN = []
keeping_track = []
special = []
index = 1
bestGuess = []
ventral = [1,1,1,-1,1,-1]
for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
    dset = data[key]['input']
    res = data[key]['analysis']
    for idn in np.sort(dset.keys())[:]:
        if idn==movingAML32:
            movIndex = index
            
        X = np.copy(dset[idn]['Neurons']['Positions']).T
        #X -=np.mean(Xref,axis=0)
        X[:,1] *=ventral[index-1]
        #X[:,1] -=np.min(X[:,1])
        
        # crop the atlas to be closer to actual size
        #Atmp = A[np.where(A[:,0]>np.min(X[:,0])*(A[:,0]<np.max(X[:,0])))]
        X = registerWorms(Xref, X, dim=3)
        # velocity/turn weights
        avWeights =res[idn][flag]['AngleVelocity']['weights']
        tWeights = res[idn][flag]['Eigenworm3']['weights']
        # list of weights for all datasets
        special.append(np.vstack([avWeights, tWeights]))
        keeping_track.append(np.ones(len(avWeights), dtype=int)*index)
        # register to atlas
        AB = registerWorms(A, X, dim=2)
        
        AN.append(AB)
        #uniquely assign IDs
        D = pairwise_distances(AB, A)
#        plt.figure()
#        plt.scatter(A[:,0],A[:,1],color=N0, s = s2, alpha=0.25, zorder=-100)#,clip_on=False)
#        plt.scatter(AB[:,0],AB[:,1],color=R0, s = s2, alpha=0.25, zorder=-100)#,clip_on=False)
        tmpID = []
        for i in range(len(AB)):
            # find best match, delete that option
            ym, xm = np.unravel_index(D.argmin(), D.shape)
            D[ym,:] = np.inf
            D[:,xm] = np.inf
            
            tmpID.append(labels[xm])
            #plt.plot(AB[ym,0],AB[ym, 1], 'x')
            #plt.plot(A[xm,0],A[xm,1], 'o')
        
        bestGuess.append(tmpID)
        index +=1
        
# all points from all datasets in one array
AB = np.concatenate(AN, axis=0)
# weights for all neurons
special = np.concatenate(special, axis=1)
# id of which dataset is which
keeping_track = np.concatenate(keeping_track, axis=0)
# replace best guess with unique assignment
bestGuess = np.concatenate(bestGuess)
# index in Atlas which neuron
AtlasLocs = np.array([np.where(labels==l)[0] for l in bestGuess ])


# subset of weighted neurons
av = np.where(np.abs(special[0])>np.percentile(np.abs(special[0]),[0]))[0]
t = np.where(np.abs(special[1])>np.percentile(np.abs(special[1]), [0]))[0]

# show atlas in lower plots
axweight4.scatter(A[:,0],A[:,1],color=N0, s = s2, alpha=0.25, zorder=-100)#,clip_on=False)
axweight3.scatter(A[:,0],A[:,1],color=N0, s = s2, alpha=0.25, zorder=-100)#,clip_on=False)


# name the most common neurons - find most common and set to zero to find next
from collections import Counter

avLabels = labels[AtlasLocs[t]]
tLabels = labels[AtlasLocs[av]]

dAV = Counter(np.ravel(AtlasLocs[av]))
dT = Counter(np.ravel(AtlasLocs[t]))

c = 30
for dic in [dAV, dT]:
    for i in range(c):
        v=list(dic.values())
        k=list(dic.keys())
        nMax = v.index(max(v))
        maxV = k[nMax]
        dic[maxV] = 0
        #axweight3.text(A[maxV,0], A[maxV,1],labels[maxV], color=R1, fontsize=6)
        print 'Rank',i, maxV, max(v), labels[maxV]
#        axweight3.scatter(A[AtlasLocs[av],0], A[AtlasLocs[av],1],marker = 'o',s=nMax, color=R1, alpha=alphaScatter)

## show all weighted labels on the atlas plots
alphaScatter = 0.3
axweight3.scatter(A[AtlasLocs[av],0], A[AtlasLocs[av],1],marker = 'o', s=s2,color=R1, alpha=alphaScatter)
alphaScatter = 0.25
axweight4.scatter(A[AtlasLocs[t],0], A[AtlasLocs[t],1],marker = 'o', s=s2,color=B1, alpha=alphaScatter)


# pritn out best ranked neurons
n = 30
most_commonAV = dAV.most_common(3)
most_commonT = dT.most_common(3)
for k, v in most_commonAV:
    print 'Rank',i, k, v, labels[k]
    
for k, v in most_commonT:
    print 'Rank',i, k, v, labels[k]
    
     
# name the ones that are already known to be motion asscociated
dAV = np.unique(AtlasLocs[av])
dT = np.unique(AtlasLocs[t])
# hardcode arrow position for motion neurons
#['AIA', 'AIB', 'AIY', 'AIZ', 'AVA', 'AVB', 'AVD', 'DVA', 'RIA', 'RIB', 'RIM', 'RIV', 'RMD', 'RME', 'SIB', 'SMB', 'SMD']
x0, y0 = 1, 1
arrows = {'AIA': (x0, 0), 'AIB':(0, -y0), 'AIY':(x0, y0), 'AIZ':(x0, -y0), 'AVA':(-x0, -y0/2.), 'AVB':(-x0, -y0), 'AVD':(0, -y0), \
    'DVA':(-x0, 0), 'RIA':(-x0, -y0), 'RIB':(-x0, 0), 'RIM':(x0, -y0), 'RIV':(-x0, 0), 'RMD':(-2*x0, y0), 'RME':(-x0, -y0/2.), 'SIB':(-x0/2., y0), 'SMB':(x0, y0), 'SMD':(0, y0)
}

l = 28
# remove all non-motion associated labels
stored = []
for loc in dAV:
    if labels[loc][:3] in motionNeurons and labels[loc][:3] not in stored:
        #axweight3.text(A[loc,0], A[loc,1],labels[loc][:-1], color=R1, fontsize=10,\
        #horizontalalignment ='center', verticalalignment='bottom')
        dx, dy = arrows[labels[loc][:3]]
        # normalize to the same length
        l0 = np.sqrt(dx**2+dy**2)
        dx *= l/l0
        dy *= l/l0
        x, y = A[loc,0], A[loc,1]
        
        axweight3.annotate(labels[loc][:-1], xy=(x, y), xytext=(x+dx, y+dy),
            arrowprops=dict(color=N0, headlength=2,headwidth=2, shrinkA=0, shrinkB=0.01,width=0.1),color=N0, fontsize=12, textcoords='data',\
            horizontalalignment='center', verticalalignment='center')
        stored.append(labels[loc][:3])

stored = []
for loc in dT:
    if labels[loc][:3] in motionNeurons and labels[loc][:3] not in stored:
        #axweight3.text(A[loc,0], A[loc,1],labels[loc][:-1], color=R1, fontsize=10,\
        #horizontalalignment ='center', verticalalignment='bottom')
        dx, dy = arrows[labels[loc][:3]]
        # normalize to the same length
        l0 = np.sqrt(dx**2+dy**2)
        dx *= l/l0
        dy *= l/l0
        x, y = A[loc,0], A[loc,1]
        
        axweight4.annotate(labels[loc][:-1], xy=(x, y), xytext=(x+dx, y+dy),
            arrowprops=dict(color=N0, headlength=2,headwidth=2, shrinkA=0, shrinkB=0.01,width=0.1),color=N0, fontsize=12, textcoords='data',\
            horizontalalignment='center', verticalalignment='center')
        stored.append(labels[loc][:3])


for ax in [axweight3, axweight4]:
    moveAxes(ax, 'scale', 0.05)
    #moveAxes(ax, 'down', 0.04)
    ax.set_xlim([-90, 136])
    ax.set_ylim([-45, 35])
    cleanAxes(ax)
    moveAxes(ax, 'left', 0.03)
    moveAxes(ax, 'down', 0.04)

# add orientation bars
length=12
xmin, ymin = axweight4.get_xlim()[1]-1.5*length, axweight4.get_ylim()[1]#-length
xmax, ymax = xmin+length, ymin+length

xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
axweight4.plot([xmid, xmid], [ymin, ymax], N0, clip_on=False)
axweight4.plot([xmin, xmax], [ymid, ymid], N0, clip_on=False)

axweight4.text(xmid, ymin, 'D', horizontalalignment = 'center', verticalalignment ='top', color=N0, fontsize=fs)
axweight4.text(xmid, ymax, 'V', horizontalalignment = 'center', verticalalignment ='bottom', color=N0, fontsize=fs)
axweight4.text(xmin, ymid, 'A', horizontalalignment = 'right', verticalalignment ='center', color=N0, fontsize=fs)
axweight4.text(xmax, ymid, 'P', horizontalalignment = 'left', verticalalignment ='center', color=N0, fontsize=fs)

# make a legend for the scatter plot
from matplotlib.lines import Line2D
leglabels = ['N = 1', 'N = 2', 'N = 3', r'N $\geq$ 4']
alphas = [alphaScatter,alphaScatter*2, alphaScatter*3, alphaScatter*4]
legend_elements = [Line2D([0], [0], marker='o', color='w', label=leglabels[i],
                          markerfacecolor=R1, alpha =alphas[i]) for i in range(4)]
leg = axweight3.legend(handles=legend_elements,  fontsize=fs, frameon = True,handletextpad =0, markerscale = 1.5,numpoints=1,\
borderaxespad=0, borderpad = 0.2, bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
legend_elements = [Line2D([0], [0], marker='o', color='w', label=leglabels[i],
                          markerfacecolor=B1, alpha =alphas[i]) for i in range(4)]
leg = axweight4.legend(handles=legend_elements, loc=4, fontsize=fs, frameon = True, handletextpad =0, markerscale = 1.5,numpoints=1,\
borderaxespad=0, borderpad = 0.2)

plt.show()

#################################################
##
##  Fourth row - pull out how much information is in different neurons by hierarchical clustering
##
#################################################
#
#gsDendro = gridspec.GridSpecFromSubplotSpec(2,3, gs1[2, :], width_ratios=[0.5,1,0.5], hspace=0.1, wspace=0.1)
#ax1 = plt.subplot(gsDendro[:2,0])
#ax2 = plt.subplot(gsDendro[:2,1])
#ax3 = plt.subplot(gsDendro[:2,2], zorder=1)
##ax4 = plt.subplot(gsDendro[3:,0])
##ax5 = plt.subplot(gsDendro[2:,1])
##ax6 = plt.subplot(gsDendro[2:,2], sharey=ax3)
#axs = [[ax1, ax2, ax3]]
#links = [L1, L2, L3, L0, N0, N1]
#set_link_color_palette(links)
#
#neurons = moving['Neurons']['RawActivity']
#t = moving['Neurons']['Time']
#print keeping_track
#labels_moving = bestGuess[np.where(keeping_track==movIndex)]
#print 'labels', len(labels_moving)
#for b, (behavior, c, lbl) in enumerate(zip(['AngleVelocity'], [R1, B1], ['Velocity', 'Turn'])):
#    beh =moving['Behavior'][behavior]
#    
#    Weights =movingAnalysis[flag][behavior]['weights']
#    
#    Relevant = np.where(np.abs(Weights>0))[0]
#    labels = labels_moving
#    
#    for li, lab in enumerate(labels):
#        if lab[:3] in motionNeurons:
#            # check if guess is good
#            if np.min(D, axis=1)[np.where(keeping_track==movIndex)][li]<5:
#                labels[li] = lab
#            else:
#                #labels[li] = "({0})".format(lab)
#                labels[li] = ''
#        else:
#            labels[li] = ''
#    if len(Relevant)<1:
#        print 'upps'
#        continue
#    
#    
#    pars = None
#    subset = Relevant
#    clust = dr.runHierarchicalClustering(moving, pars, subset)
#    
#    dn = dendrogram(clust['linkage'],ax = axs[b][0],leaf_font_size=10, leaf_rotation=0,\
#         orientation = 'left', show_leaf_counts=1, above_threshold_color='k',labels = labels, color_threshold= clust['threshold'])
#   
#    xlbls = axs[b][0].get_ymajorticklabels()
#    
#    for lbi,lb in enumerate(xlbls):
#        lb.set_color(links[clust['leafs'][dn['leaves'][lbi]]-1])
##    
#    traces = clust['clusters']
#    
#    for n in range( clust['nclusters']):
#        
#        scale = 1.5
#        axs[b][1].plot(t, traces[n].T+scale*n, 'k', alpha=0.2)
#        axs[b][1].plot(t, np.nanmean(traces[n], axis=0)+n*scale, color= links[n])
#        # sort by behavior and plot
#        scale = 1
#        xPlot, avg, std = sortnbin(beh, np.nanmean(traces[n], axis=0), nBins=10, rng=(np.min(beh), np.max(beh)))
#        # sort each neuron and average after
#        #avg = np.mean([sortnbin(beh, trc, nBins=5, rng=(np.min(beh), np.max(beh)))[1] for trc in traces[n]], axis=0)
#        #std = np.std([sortnbin(beh, trc, nBins=5, rng=(np.min(beh), np.max(beh)))[1] for trc in traces[n]], axis=0)
#        axs[b][2].plot(xPlot, avg+n*scale, color= links[n])
#        axs[b][2].fill_between(xPlot, avg-std+n*scale, avg+std+n*scale,color= links[n], alpha=0.5)
#        axs[b][2].plot(xPlot,n*scale*np.ones(len(xPlot)), color= 'k', linestyle=':')
#        #dashed line at zero
#        axs[b][2].axvline(0,0, 1, color= 'k', linestyle=':')
#        if b==0:
#            axs[b][2].axvspan(-0.05, 0.05, zorder=-10, alpha=0.1, color='k')
#        if b==1:
#            axs[b][2].axvspan(-3, 3, zorder=-10, alpha=0.1, color='k')
#            
#        
#    # scale bar
#    if b==0:
#        axs[0][2].plot([xPlot[0], xPlot[0]], [(n+0.5)*scale,(n+0.5)*scale+0.5], color='k', lw = 2)
#        axs[0][2].text(xPlot[0]+0.02,(n+0.5)*scale , r'$\Delta R/ R_0 = 0.5$', verticalalignment = 'bottom')#, transform = axs[0][2].transAxes)
#        
##    axs[1][2].set_xticks([-10, 10])
##    axs[1][2].set_xticklabels(['Ventral',' Dorsal'])
#    axs[0][2].set_xticks([-0.2, 0.4])
#    axs[0][2].set_xticklabels(['Reverse',' Forward'])
#        
#    axs[b][2].spines['left'].set_visible(False)
#    #axs[b][2].set_xlabel(lbl)
#    
#    axs[b][2].set_yticks([])
#    axs[0][1].set_xticks([])
#    axs[b][0].spines['left'].set_visible(False)
#    axs[b][0].set_ylabel(lbl)
#    axs[b][1].spines['left'].set_visible(False)
#    axs[0][1].spines['bottom'].set_visible(False)
#   
#    axs[b][1].set_yticks([])
##    axs[1][0].spines['bottom'].set_visible(False)
#    axs[0][0].spines['bottom'].set_visible(False)
#    axs[0][0].set_xticks([])
##    axs[1][1].set_xlabel('Time (s)')
#    
#
#moveAxes(ax1, 'left', 0.05)
#for ax in [ax1, ax2, ax3]:
#    #moveAxes(ax, 'scale', 0.025)
#    moveAxes(ax, 'up', 0.05)
#for ax in [axweight3, axweight4]:
#    #moveAxes(ax, 'scale', 0.025)
#    moveAxes(ax, 'down', 0.02)
#plt.show()
#
#
##ax = plt.subplot(gs1[4,0])
##for ind, y in enumerate(interestingNeurons):
##    #y = interestingNeurons[neur] 
##    ax.plot(ind+y[1][:1500], 'k', lw = 1)
##    ax.plot(ind+y[2][:1500]*10, 'k', lw = 1)
##    #plotEthogram(ax, np.arange(len(y[0]))[:100], y[1][:100], alpha = 1, yValMax=ind+1, yValMin=ind-1, legend=0)
##    #plt.show()
##    ax.text(-50, ind, y[0])
##ax.set_ylim(-4, len(interestingNeurons))
#
#
## plot activity as a function of velocity and turns
#bins = np.arange(-0.05, 0.06, 0.005)
#ybins = np.arange(-0.5, 1.5, 0.01)
#gs = gridspec.GridSpecFromSubplotSpec(1,8, gs1[3, :])
#for ind, y in enumerate(interestingNeurons[:8]):
#    ax = plt.subplot(gs[ind])
#    
#    H, xe, ye = np.histogram2d(y[2], y[1], (bins, ybins))
#    
#    m, s = np.nanmean(H*(ye[:-1]+np.diff(ye)[0]*0.5), axis=1), np.nanstd(H, axis=1)/np.sqrt(np.nansum(H, axis=1))
#    ax.plot(xe[:-1]+np.diff(xe)[0]*0.5, m, label = y[0])
#    ax.fill_between(xe[:-1]+np.diff(xe)[0]*0.5, m-s,m+s, alpha=0.5)
#    ax.legend()
##    ax.scatter(np.sort(y[1]), y[0][np.argsort(y[1])])
#
#
#plt.show()


## predict immobile behavior
#
## load immobile worm from fig1
## select a special dataset - transiently immobilized
#transient = data['AML32_chip']['input']['BrainScanner20180511_134913']
#transientAnalysis = data['AML32_chip']['analysis']['BrainScanner20180511_134913']
## time first half, second half. Indices of times
#timeHalf = np.arange(0, 1400)
#time2Half = np.arange(1600, transient['Neurons']['Activity'].shape[1])
## pull out repeated stuff
#time = transient['Neurons']['TimeFull']
#timeActual = transient['Neurons']['Time']
#noNeurons = transient['Neurons']['Activity'].shape[0]
##
#label = 'AngleVelocity'
#splits = transientAnalysis['Training']
#train, test = splits[label]['Train'], splits[label]['Test']
#t = time[test]
#axImm = plt.subplot(gs1[3, 0])
#
#ethoImm = dh.makeEthogram(transientAnalysis[flag]['AngleVelocity']['output'][test], transientAnalysis[flag]['Eigenworm3']['output'][test])
## plot predicted behavior
#for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
#            [N1, N1], [R1, B1],[0, 1], ['Wave speed', 'Turn']):
#    #beh = transient['Behavior'][behavior][test]
#    #meanb, maxb = np.mean(beh),np.std(beh)
#    #beh = (beh-meanb)/maxb
#    #beh*=scale/10
#    behPred = transientAnalysis[flag][behavior]['output'][test]
#    #behPred = (behPred-np.mean(behPred))#/np.max(behPred)
#    print transientAnalysis[flag][behavior]['score']
#    print np.max(behPred)
#    #behPred*=scale/10
#    #axImm.plot(t, beh+yl, color=color)
#    #axImm.plot(t, behPred+yl, color=cpred)
#    axImm.text(t[-1], yl+scale/5, \
#    r'$R^2 = {:.2f}$'.format(np.float(transientAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
#    axImm.text(t[-1]*1.1, yl, label, rotation=90, color=cpred, verticalalignment='center')
#    plotEthogram(axImm, t, ethoImm, alpha = 0.5, yValMax=1, yValMin=0, legend=0)