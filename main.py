#!/usr/bin/env python
# coding: utf-8

# In[1]:

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    import cv2,random,pickle
    plt.rcParams["axes.grid"] = False
    import tensorflow as tf
    import keras
    from keras.models import Model, Sequential,load_model
    import h5py


    # In[2]:


    from data import get_data_raw,get_data_resized
    from get_model import get_models
    from get_display import get_matching
    from get_cluster_number import get_number_of_clusters
    from get_visualize_clusters import visual_cluster


    # In[3]:


    master_list=get_data_raw()


    # In[4]:


    img_resized=get_data_resized(master_list)
    #save_it(img_resized)


    # In[5]:


    '''with open('img_resized.pkl','rb') as f:
        img_resized = pickle.load(f)'''


    # In[6]:


    autoencoder,encoder=get_models()


    # In[7]:


    pretrain_epochs = 60
    batch_size = 16
    #autoencoder.fit(img_resized, img_resized, batch_size=batch_size, epochs=pretrain_epochs,verbose=1)
    #We have already trained


    # In[8]:


    autoencoder.load_weights('conv_ae_weights_final_deployv7_img_160 .h5')


    # In[10]:
    encoder_output=encoder.predict(img_resized)
    '''with open('encoder_output.pkl', 'wb') as f:
    pickle.dump(encoder_output, f)'''
    
    '''with open('encoder_output.pkl','rb') as f:
    encoder_output = pickle.load(f)'''

    # In[11]:

    get_matching(encoder_output,img_resized)
    #get_number_of_clusters(encoder_output)





    # In[14]:


    print('Do you want to view the clusters?')
    print('Enter Yes or No')
    choice=input()
    if(choice=='Yes'):
        print('How many items do you want to view per cluster?')
        num=int(input())
        n_clusters=15
        visual_cluster(encoder_output,img_resized,n_clusters,num)


if __name__=='__main__':
    main()
