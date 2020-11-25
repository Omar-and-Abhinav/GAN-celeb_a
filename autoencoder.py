import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras import Input
import numpy as np
import matplotlib as plt
from tensorflow import keras 

dataset = photo_dataset()
dataset.load_data(10000)
x_train, x_val, x_test = dataset.split_data(dataset.data, 0.8, 0.0)




def autoencoder_func():
    
    # Encoder
    inp = Input(shape = (109, 89, 3), name = 'input')
    x = Conv2D(64, (3, 3), activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    #Decoder
    x = Conv2DTranspose(16, (3, 3), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32, (3, 3), activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
    out = Conv2DTranspose(3, (4, 4), activation = 'relu')(x)
    
    autoencoder = keras.Model(inputs = inp, outputs = out)
    autoencoder.summary()
    
    autoencoder.compile(
        optimizer = keras.optimizers.Adam(0.01),
        loss = 'mean_squared_error',
        metrics = ['accuracy'])
    autoencoder.fit(x_train, x_train, epochs=10, batch_size=10, verbose=2)
    
    
    