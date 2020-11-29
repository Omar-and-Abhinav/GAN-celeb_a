import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os

'''dataset = photo_dataset()
dataset.load_data(30000)
x_train, x_val, x_test = dataset.split_data(dataset.data, 0.8, 0.0)'''

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
    

# GENERATOR    
gen_inp = Input(shape = (9, 7, 8), name = 'gen_noise')
x = Conv2DTranspose(6, (3, 3), activation = 'relu', name = 'covtr1')(gen_inp)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(12, (3, 3), activation = 'relu', name = 'covtr2')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(24, (3, 3), activation = 'relu', name = 'covtr3')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
gen_out = Conv2DTranspose(3, (10, 6), activation='sigmoid', name = 'covtr4')(x)

generator = keras.Model(inputs = gen_inp, outputs = gen_out, name = 'Generator' )
generator.summary()

# DISCRIMINATOR
disc_inp = Input(shape = (109, 89, 3), name = 'disc_inp')
x = Conv2D(32, (4, 4), strides = (3, 3), activation = 'relu', name = 'firstDiscLayer')(disc_inp)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), strides = (2, 2) , activation = 'relu', name='secondDiscLayer' )(x)
x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), strides = (2, 2), activation = 'relu', name = 'thirdDiscLayer')(x)
x = BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
disc_out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

discriminator = keras.Model(inputs = disc_inp, outputs = disc_out, name = 'Discriminator')
discriminator.summary()

