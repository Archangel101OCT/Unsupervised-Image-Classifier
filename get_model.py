#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import keras
import keras.backend as K
from keras.models import Model, Sequential,load_model
from keras.layers import BatchNormalization,Dropout,MaxPooling2D,UpSampling2D,Dense,Input,Conv2D,Activation,Flatten,Conv2DTranspose,Lambda,Reshape,Layer,InputSpec


# In[4]:


img_size=160


# In[5]:


def autoencoderConv2D_2(img_shape=(160, 160, 3)):

    input_img = Input(shape=img_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x=MaxPooling2D((2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x=MaxPooling2D((2,2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x=MaxPooling2D((2,2))(x)
 
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    encoded=Dense(150,activation='relu')(x)#100
    x=BatchNormalization()(encoded)
    
    
    # Decoder
    # Reshape into an image of the same shape as before our last `Flatten` layer
    x=Dense(np.product(shape_before_flattening[1:]),activation='relu')(x)
    x=BatchNormalization()(x)
    x = Reshape(shape_before_flattening[1:])(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)



    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    '''x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)'''
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


def get_models():
    autoencoder,encoder=autoencoderConv2D_2()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return autoencoder,encoder