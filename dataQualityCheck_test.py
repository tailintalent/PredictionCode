
# coding: utf-8

# In[2]:


# standard modules
import matplotlib as mpl
import numpy as np
import matplotlib.pylab as plt
import h5py
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr
from util import get_label_dict, filter_filename, sort_two_lists
import pandas as pd
from collections import Counter
import pickle


# In[ ]:


mpl.rcParams['interactive']  = False
###############################################    
# 
#    run parameters
#
###############################################
typ = 'AML32' # possible values AML32, AML18, AML70
condition = 'moving'# # Moving, immobilized, chip
first = True # if true, create new HDF5 file
###############################################    
# 
#    load data into dictionary
#
##############################################
folder = '{}_{}/'.format(typ, condition)
dataLog = '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
outLoc = "Analysis/{}_{}_results.hdf5".format(typ, condition)
outLocData = "Analysis/{}_{}.hdf5".format(typ, condition)

# data parameters
dataPars = {'medianWindow':50, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':50, # gauss window for angle velocity derivative. Acts on full (50Hz) data
            'rotate':False, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 6,  # gauss window for red and green channel
            'interpolateNans': 6,#interpolate gaps smaller than this of nan values in calcium data
            }

# Use the 6 moving worms:
filename = 'datasets/AML70_chip.hdf5'
dataSets = dict(h5py.File(filename, 'r'))
filename = 'datasets/AML32_moving.hdf5'
dataSets2 = dict(h5py.File(filename, 'r'))
dataSets.update(dataSets2)


# dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, dataPars = dataPars, nDatasets = None)
keyList = np.sort(list(dataSets.keys()))
    
print(keyList)
keyList = keyList
# results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
# analysis parameters

pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':False, #timewarp so behaviors are equally represented
        'trainingCut': 0.6, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
        'useDeconv': 0, # use the deconvolved transformed version of neural data for all analyses
        'nCluster': 10, # use the deconvolved transformed version of neural data for all analyses
        'useClust':False,# use clusters in the fitting procedure.
        'useDeriv':False,# use neural activity derivative for PCA
        'useRaw':False,# use neural R/R0
        'testVolumes' : 6*60*1, # 2 min of data for test sets in nested validation
        'periods': np.arange(0, 300) # relevant periods in seconds for timescale estimate
        
     }

behaviors = ['AngleVelocity','Eigenworm3']
#behaviors = ['Eigenworm3']

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = 1#True 
overview = 0#False
predNeur = 0
predPCA = 0
bta = 0
svm = 0
pca = 0#False
kato_pca= 0
half_pca= 0
hierclust = False
linreg = False
periodogram = 0
nestedvalidation = 0
lasso = 0
elasticnet = 1#True
lagregression = 0
positionweights = 0#True
resultsPredictionOverview = 1
transient = 0
###############################################    
# 
# create training and test set indices
# 
##############################################
if createIndicesTest:
    for kindex, key in enumerate(keyList):
        resultDict[key] = {'Training':{}}
        for label in behaviors:
            train, test = dr.createTrainingTestIndices(dataSets[key], pars, label=label)
            if transient:
               train = np.where(dataSets[key]['Neurons']['Time']<4*60)[0]
                # after 4:30 min
               test = np.where((dataSets[key]['Neurons']['Time']>7*60)*(dataSets[key]['Neurons']['Time']<14*60))[0]
               resultDict[key]['Training']['Half'] ={'Train':train}
               resultDict[key]['Training']['Half']['Test'] = test
            else:
                 # add half split
                midpoint = np.mean(dataSets[key]['Neurons']['Time'])
                trainhalf = np.where(dataSets[key]['Neurons']['Time']<midpoint)[0]
                testhalf = np.where(dataSets[key]['Neurons']['Time']>midpoint)[0]
                resultDict[key]['Training']['Half'] ={'Train':trainhalf}
                resultDict[key]['Training']['Half']['Test'] = testhalf
            resultDict[key]['Training'][label] = {'Train':train  }
            resultDict[key]['Training'][label]['Test']=test
           

    print("Done generating trainingsets")


###############################################    
# 
# linear regression using elastic Net
#
##############################################
if elasticnet:
    for kindex, key in enumerate(keyList):
        print('Running Elastic Net',  key)
        splits = resultDict[key]['Training']
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars,splits, plot=1, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], splits,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey].update(tmpDict[tmpKey])
            
#         # do converse calculation -- give it only the neurons non-zero in previous case
#         subset = {}
#         subset['AngleVelocity'] = np.where(np.abs(resultDict[key]['ElasticNet']['Eigenworm3']['weights'])>0)[0]
#         subset['Eigenworm3'] = np.where(np.abs(resultDict[key]['ElasticNet']['AngleVelocity']['weights'])>0)[0]
#         resultDict[key]['ElasticNet']['ConversePrediction'] = dr.runLinearModel(dataSets[key], resultDict[key], pars, splits, plot = True, behaviors = ['AngleVelocity', 'Eigenworm3'], fitmethod = 'ElasticNet', subset = subset)
        
#     mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='ElasticNet', behaviors = behaviors,random = pars['trainingType'])
#     plt.show(block=True)

# pickle.dump(resultDict, open("resultDict_6.p", "wb"))


# In[9]:


# resultDict = pickle.load(open("resultDict_6.p", 'rb'))

non_zero_weights_list = []
# behavior = 'Eigenworm3'
behavior = 'AngleVelocity'
print("Behavior: {0}".format(behavior))
for key, item in resultDict.items():
    print(key)
    weights = resultDict[key]['ElasticNet'][behavior]['weights']
    # Obtain id-neuron correspondence for each dataset:
    label_dict, label_inverse_dict = get_label_dict(key)
    # Obtain neuron names with non-zero weights:
    non_zero_weights = [label_dict[id] for id, weight in enumerate(weights) if np.abs(weight) > 0]
    non_zero_weights_list = non_zero_weights_list + non_zero_weights
    
# Count the accumulated number nonzero weights for each neuron:
nonzero_counter = Counter(non_zero_weights_list)
nonzero_keys, nonzero_values = list(nonzero_counter.keys()), list(nonzero_counter.values())
nonzero_values_sorted, nonzero_keys_sorted = sort_two_lists(nonzero_values, nonzero_keys, reverse = True)
d = {'Velocity neurons': nonzero_keys_sorted, 'Number of recordings': nonzero_values_sorted}
df = pd.DataFrame(data=d)
df 


# In[7]:


from collections import Counter
non_zero_weights_list = []
behavior = 'Eigenworm3'
# behavior = 'AngleVelocity'
print("Behavior: {0}".format(behavior))
for key, item in resultDict.items():
    print(key)
    weights = resultDict[key]['ElasticNet'][behavior]['weights']
    # Obtain id-neuron correspondence for each dataset:
    label_dict, label_inverse_dict = get_label_dict(key)
    # Obtain neuron names with non-zero weights:
    non_zero_weights = [label_dict[id] for id, weight in enumerate(weights) if np.abs(weight) > 0]
    non_zero_weights_list = non_zero_weights_list + non_zero_weights
    
# Count the accumulated number nonzero weights for each neuron:
nonzero_counter = Counter(non_zero_weights_list)
nonzero_keys, nonzero_values = list(nonzero_counter.keys()), list(nonzero_counter.values())
nonzero_values_sorted, nonzero_keys_sorted = sort_two_lists(nonzero_values, nonzero_keys, reverse = True)
d = {'turn neurons': nonzero_keys_sorted, 'Number of recordings': nonzero_values_sorted}
df = pd.DataFrame(data=d)
df 

