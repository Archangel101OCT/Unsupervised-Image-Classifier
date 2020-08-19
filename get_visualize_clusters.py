#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()
plt.rcParams["axes.grid"] = False


def visual_cluster(encoder_output,img_resized,n_clusters=15,num=5):
    kmeans=KMeans(n_clusters)
    yy=kmeans.fit(encoder_output)
    pred_y=yy.labels_
    clusters={}
    for i in np.unique(pred_y):
        selection=img_resized[pred_y==i]
        clusters[i]=selection
    clusters_are =np.unique(pred_y)  #enter label number to visualise
    for clust in clusters_are:
        plt.figure(figsize=(20,20))
        print('---------------Cluster Number ',clust) 
        for i in range(1,num+1):
            plt.subplot(10, 10, i); #(Number of rows, Number of column per row, item number)
            plt.imshow(clusters[clust][(i+50)%len(clusters[clust])]);
        plt.show()
