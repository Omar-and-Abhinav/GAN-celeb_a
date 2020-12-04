import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import os

loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

dataset = photo_dataset()
dataset.load_data(10000)
#dataset.data = dataset.data/255
#x_train, x_val, x_test = dataset.split_data(dataset.data, 0.8, 0.0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    

# def discriminator_loss(real_output, fake_output):
#     # f1 = -tf.math.multiply(tf.math.log(y_pred), y_true)
#     # f2 = -tf.math.multiply(tf.math.log(tf.ones(y_pred.shape[0]) - y_pred), tf.ones(y_pred.shape[0]) - y_true)
#     # return(tf.math.reduce_mean(tf.add(f1, f2)))
#     # return(f1, f2)
#     real_loss=loss_func(tf.ones_like(real_output), real_output)
#     fake_loss=loss_func(tf.zeros_like(fake_output), fake_output)
#     return real_loss+fake_loss

def generator_loss(fake_output):
    return loss_func(tf.ones_like(fake_output), fake_output)

init = tf.keras.initializers.RandomNormal(stddev=0.02, mean = 0.0)
# GENERATOR    
gen_inp = Input(shape = (100,), name = 'gen_noise')
x = keras.layers.Dense(13*11*3, activation = 'relu', name = 'covtr1')(gen_inp)
x = keras.layers.Reshape((13, 11, 3))(x)
x = Conv2DTranspose(filters = 16,kernel_size =  (4,4), strides = (2, 2), padding = 'same',  name = 'covtr2', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Conv2DTranspose(filters = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'same', name = 'covtr3', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Conv2DTranspose(filters = 64,kernel_size =  (3,3), strides = (2, 2), padding = 'same',  name = 'covtr4', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization(momentum=0.8)(x)
gen_out = Conv2DTranspose(3, (6, 2), activation='linear', name = 'covtr5', kernel_initializer=init)(x)

generator = keras.Model(inputs = gen_inp, outputs = gen_out, name = 'Generator' )
generator.summary()


# DISCRIMINATOR
disc_inp = Input(shape = (109, 89, 3), name = 'disc_inp')
x = Conv2D(128, (4, 4), strides = (3, 3), name = 'firstDiscLayer', kernel_initializer=init)(disc_inp)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), strides = (2, 2) , name='secondDiscLayer' , kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), strides = (2, 2), name = 'thirdDiscLayer', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization(momentum=0.8)(x)
x = Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
disc_out = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer=init)(x)

discriminator = keras.Model(inputs = disc_inp, outputs = disc_out, name = 'Discriminator')
discriminator.summary()

d_optimizer = keras.optimizers.Adam(1.5e-4,0.5)
g_optimizer = keras.optimizers.Adam(1.5e-4,0.5)
latent_dim = (100,)

@tf.function
def train_gans(data, train_real = True):
        data = tf.cast(data, tf.float32)
        data = tf.reshape(data, (-1, 109, 89, 3))
        random_latent = tf.random.normal((batch_size,) + latent_dim)
        # images = tf.concat([generated_images, data], axis=0)
        # labels = tf.concat(
        #     [tf.zeros(batch_size, 1), tf.ones(batch_size, 1)], axis = 0)
        # 
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = generator(random_latent, training = True)
            if train_real:
                predictions = discriminator(data, training = True)
                labels = tf.ones((batch_size, 1))
                d_loss = loss_func(labels, predictions)
            else:
                predictions = discriminator(generated_images, training = True)
                labels = tf.zeros((batch_size, 1))
                d_loss = loss_func(labels, predictions)
            fake_output = discriminator(generated_images, training = True)
            g_loss = generator_loss(fake_output)
            
            disc_grad = disc_tape.gradient(d_loss, discriminator.trainable_weights)
            gen_grad = gen_tape.gradient(g_loss, generator.trainable_variables)
            d_optimizer.apply_gradients(zip(disc_grad , discriminator.trainable_weights))
            g_optimizer.apply_gradients(zip(gen_grad , generator.trainable_weights))
        return g_loss, d_loss



epochs = 50
batch_size = 20
split_factor = dataset.data.shape[0]/batch_size
for epoch in range(epochs):
    print("Epoch Number", epoch + 1, end = ' ')
    d_losses = []
    g_losses = []
    for step, real_images in enumerate(np.split(dataset.data, split_factor)):
                if step%2==0:
                    losses = train_gans(real_images)
                    d_losses.append(losses[1])
                    g_losses.append(losses[0])
                else:
                    losses = train_gans(real_images, False)
                    d_losses.append(losses[1])
                    g_losses.append(losses[0])
    print('d_loss:', np.mean(d_losses), 'g_loss:', np.mean(g_losses))
    del d_losses, g_losses

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
