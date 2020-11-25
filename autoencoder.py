" # CODE FOR AUTOENCODER # "  

import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras import Input
import numpy as np
import matplotlib as plt
from tensorflow import keras 

dataset = photo_dataset()
dataset.load_data(300)
x_train, x_val, x_test = dataset.split_data(dataset.data, 0.8, 0.1)


inp = Input(shape = ( 109, 89, 3), name = 'input')
x = Conv2D( 16, strides = (3, 3), activation='relu')(inp)
x = Conv2D( 8, strides = (3, 3), activation='relu')(x)
x = Conv2D( 4, strides = (3, 3), activation='relu')(x)
x = Conv2DTranspose(4, strides = (3, 3), activation = 'relu')(x)
x = Conv2DTranspose(8, strides = (3, 3), activation = 'relu')(x)
out = Conv2DTranspose(16, strides = (3, 3), activation = 'relu')(x)

def autoencoder(data):
    
    inp=data
    inp = Input(shape = ( 109, 89, 3), name = 'input')
    x = Conv2D( 16, strides = (3, 3), activation='relu')(inp)
    x = Conv2D( 8, strides = (3, 3), activation='relu')(x)  
    x = Conv2D( 4, strides = (3, 3), activation='relu')(x)
    x = Conv2DTranspose(4, strides = (3, 3), activation = 'relu')(x)
    x = Conv2DTranspose(8, strides = (3, 3), activation = 'relu')(x)
    out = Conv2DTranspose(16, strides = (3, 3), activation = 'relu')(x)
    