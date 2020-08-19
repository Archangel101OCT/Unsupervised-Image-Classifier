#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams["axes.grid"] = False
from sklearn.metrics.pairwise import cosine_similarity


# In[6]:


def get_matching(encoder_output,img_resized):
    img_size=160
    print('Input an image id which you want to find similar images of')
    print('Note image id are from 0 to 4737')
    print('Enter -1 to quit')
    n=int(input())
    while(n>=0 & n<=4737):
        plt.show()
        enc=np.array(encoder_output).reshape(encoder_output.shape[0],encoder_output.shape[1])
        ref=[]
        for i in range(len(enc)):
            temp=enc[i].reshape(150,1).T
            ref.append(temp)

        dist={}
        for i in range(len(ref)):
            if(i!=n):
                kk=cosine_similarity(ref[n],ref[i])
                dist[i]=kk
        dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        idex=dist
        no_of_channels=3

        plt.imshow(img_resized[n].reshape(img_size,img_size,no_of_channels),cmap='gray')

        fig, axes = plt.subplots(3, 3,figsize=(15,15))
        axes[0,0].imshow(img_resized[idex[0][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        #ax1.set_title('Orignal')
        axes[0,1].imshow(img_resized[idex[1][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        #ax2.set_title('PREDICTED')
        axes[0,2].imshow(img_resized[idex[2][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')    

        axes[1,0].imshow(img_resized[idex[3][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        axes[1,1].imshow(img_resized[idex[4][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        axes[1,2].imshow(img_resized[idex[5][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')

        axes[2,0].imshow(img_resized[idex[6][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        axes[2,1].imshow(img_resized[idex[7][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        axes[2,2].imshow(img_resized[idex[8][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
        plt.show()
        n=int(input())