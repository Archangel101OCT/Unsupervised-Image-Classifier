#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
sns.set()
plt.rcParams["axes.grid"] = False
from sklearn.cluster import KMeans



# In[2]:


def get_number_of_clusters(encoder_output):
    distortions = [] 
    inertias = [] 
    mapping1 = {} 
    mapping2 = {} 
    K = range(1,20) 

    for k in K:
        print(k,end=' ')
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k).fit(encoder_output) 
        kmeanModel.fit(encoder_output)     

        distortions.append(sum(np.min(cdist(encoder_output, kmeanModel.cluster_centers_, 
                          'euclidean'),axis=1)) / encoder_output.shape[0]) 
        inertias.append(kmeanModel.inertia_) 

        mapping1[k] = sum(np.min(cdist(encoder_output, kmeanModel.cluster_centers_, 
                     'euclidean'),axis=1)) / encoder_output.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 


    plt.plot(K, distortions, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Distortion') 
    plt.title('The Elbow Method using Distortion') 
    plt.show() 
    plt.plot(K, inertias, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia') 
    plt.title('The Elbow Method using Inertia') 
    plt.show()
