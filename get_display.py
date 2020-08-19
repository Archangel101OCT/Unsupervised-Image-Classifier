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
                kk=cosine_similarity(ref[n],ref[i])  #Using cosine similarity to get nearest images
                dist[i]=kk
        dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

        idex=dist
        no_of_channels=3
        print('Enter the number of n nearest images that you want to see')
        k=int(input())
        plt.imshow(img_resized[n].reshape(img_size,img_size,no_of_channels),cmap='gray')
        if(k%3!=0):
            rw=int(k/3)+1
        else:
            rw=k/3
        
        c=0
        count=0
        if(k<=3):
            fig, axes = plt.subplots(1,3,figsize=(15,15))
            for i in range(1):
                axes[0].imshow(img_resized[idex[c][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
                #ax1.set_title('Orignal')
                c=c+1
                count=count+1
                if(count>=k):
                    break
                axes[1].imshow(img_resized[idex[c][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
                #ax2.set_title('PREDICTED')
                c=c+1
                count=count+1
                if(count>=k):
                    break
                axes[2].imshow(img_resized[idex[c][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')    
                c=c+1
                count=count+1
                if(count>=k):
                    break
        else:   
            fig, axes = plt.subplots(rw,3,figsize=(15,15))
            for i in range(rw):
                axes[i,0].imshow(img_resized[idex[c][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
                #ax1.set_title('Orignal')
                c=c+1
                count=count+1
                if(count>=k):
                    break
                axes[i,1].imshow(img_resized[idex[c][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')
                #ax2.set_title('PREDICTED')
                c=c+1
                count=count+1
                if(count>=k):
                    break
                axes[i,2].imshow(img_resized[idex[c][0]].reshape(img_size,img_size,no_of_channels),cmap='gray')    
                c=c+1
                count=count+1
                if(count>=k):
                    break
        plt.show()
        n=int(input())