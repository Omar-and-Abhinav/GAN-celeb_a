import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os

dataset = photo_dataset()
dataset.load_data(30000)
#x_train, x_val, x_test = dataset.split_data(dataset.data, 0.8, 0.0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    

def discriminator_loss(y_true, y_pred):
    f1 = -tf.math.multiply(tf.math.log(y_pred), y_true)
    f2 = -tf.math.multiply(tf.math.log(tf.ones(y_pred.shape[0]) - y_pred), tf.ones(y_pred.shape[0]) - y_true)
    return(tf.math.reduce_mean(tf.add(f1, f2)))
    #return(f1, f2)
    

def generator_loss(y_pred):
    return -tf.math.reduce_mean(tf.math.log(y_pred))


class GAN(tf.keras.Model):
    
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
    
    def compile(self, d_optimizer, g_optimizer, d_loss, g_loss):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss
    
    def train_step(self, data):
        data = tf.cast(data, tf.float32)
        batch_size = tf.shape(data)[0]
        
        #  DISCRIMINATOR
        random_latent = tf.random.normal((batch_size,) + self.latent_dim)
        generated_images = self.generator(random_latent)
        images = tf.concat([generated_images, data], axis=0)
        labels = tf.concat(
            [tf.zeros(batch_size, 1), tf.ones(batch_size, 1)], axis = 0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        with tf.GradientTape() as tape:
            predictions = self.discriminator(images)
            print("Shape:", predictions.shape, labels.shape, images.shape)
            d_loss = self.d_loss(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        #  GENERATOR
        random_latent = tf.random.normal((batch_size,) + self.latent_dim)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent))
            g_loss = self.g_loss(predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}
            

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
gen_out = Conv2DTranspose(3, (10, 6), activation='tanh', name = 'covtr4')(x)

generator = keras.Model(inputs = gen_inp, outputs = gen_out, name = 'Generator' )
#generator.summary()

# DISCRIMINATOR
disc_inp = Input(shape = (109, 89, 3), name = 'disc_inp')
x = Conv2D(32, (4, 4), strides = (3, 3), name = 'firstDiscLayer')(disc_inp)
x = LeakyReLU(alpha = 0.3)(x)
x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), strides = (2, 2) , name='secondDiscLayer' )(x)
x = LeakyReLU(alpha = 0.3)(x)
x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), strides = (2, 2), name = 'thirdDiscLayer')(x)
x = LeakyReLU(alpha = 0.3)(x)
x = BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
disc_out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

discriminator = keras.Model(inputs = disc_inp, outputs = disc_out, name = 'Discriminator')
#discriminator.summary()



gan = GAN(discriminator, generator, (9, 7, 8))
gan.compile(
    keras.optimizers.Adam(learning_rate = 0.003),
    keras.optimizers.Adam(learning_rate = 0.003),
    discriminator_loss,
    generator_loss
)
gan.fit(dataset.data/255, epochs = 10, batch_size = 10)


def show_images(images, cols = 1, titles = None):
    """From https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

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