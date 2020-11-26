import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os.path

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




    
# Encoder
inp = Input(shape = (109, 89, 3), name = 'input')
x = Conv2D(32, (3, 3), activation='relu')(inp)
#x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D((2,2))(x)
x = Conv2D(8, (3, 3), activation='relu')(x)
#Decoder
x = Conv2DTranspose(8, (3, 3), activation = 'relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(16, (3, 3), activation = 'relu')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
out = Conv2DTranspose(3, (4, 4), activation = 'relu')(x)

autoencoder = keras.Model(inputs = inp, outputs = out)
autoencoder.summary()

def run(x_train, valid_split, epochs, batch):
    autoencoder.compile(
    optimizer = keras.optimizers.Adam(0.001),
    loss = 'mean_squared_error',
    metrics = ['accuracy'])
    autoencoder.fit(x_train, x_train, 
                    validation_split = valid_split, 
                    epochs = epochs, 
                    batch_size = batch)

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


