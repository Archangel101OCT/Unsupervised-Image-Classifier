#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import os,random
import pickle

# In[2]:


def get_data_raw():
    print('Getting master data list, please be patient')
    path='dataset/'
    kt=os.listdir(path)
    for i in range(len(kt)):
        kt[i]=int(kt[i][:-4])
    kt.sort()
    for i in range(len(kt)):
        kt[i]=str(kt[i])+'.jpg'
    images_master=[]
    for imdr in kt:
        image=cv2.imread(os.path.join(path,imdr))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_master.append(image)
    return images_master


# In[3]:


def get_data_resized(images_master):
    print('Preprocessing data(may take some time)')
    images_temp=images_master.copy()
    img_size=160
    img_resized=[]
    for img in images_temp:
        img=img.astype('float32')/255
        img=cv2.resize(img,(img_size,img_size))
        img_resized.append(img)
    img_resized=np.asarray(img_resized)
    #img_resized=np.array(img_resized).reshape(-1,img_size,img_size,3)
    return img_resized

def save_it(img_resized):
    with open('img_resized.pkl','wb') as f:
        pickle.dump(img_resized, f)
