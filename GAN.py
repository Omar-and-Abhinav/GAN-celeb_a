import tensorflow as tf
from load_celeb_data import photo_dataset
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import tensorflow.keras.backend as K

K.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


dataset = photo_dataset("celeb_a_updated")
dataset.load_data(150000, scale = 1)
d_optimizer = keras.optimizers.Adam(0.0001,0.5)
g_optimizer = keras.optimizers.Adam(0.0001,0.5)
latent_dim = (15*5*3,)
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
init = tf.keras.initializers.RandomNormal(stddev=0.02, mean=0.0)
save_modelandimages = 'FinalModel20'
input_shape = (55, 45, 3)

# GENERATOR    
gen_inp = Input(shape = latent_dim, name = 'gen_noise')
x = keras.layers.Reshape((15, 5, 3))(gen_inp)
x = Conv2DTranspose(filters = 4, kernel_size =  (2, 2), padding = 'valid',  name = 'covtr4', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 6, kernel_size =  (3, 3), padding = 'valid',  name = 'covtr5', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 8, kernel_size =  (4, 4), padding = 'valid',  name = 'covtr6', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 16, kernel_size =  (5, 5), padding = 'valid',  name = 'covtr7', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 20, kernel_size =  (6, 6), padding = 'valid',  name = 'covtr8', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 24, kernel_size =  (7, 7), padding = 'valid',  name = 'covtr9', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 30, kernel_size =  (8, 8), padding = 'valid',  name = 'covtr10', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(filters = 34, kernel_size =  (9, 9), padding = 'valid',  name = 'covtr11', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)

gen_out = Conv2DTranspose(3, (5, 5), activation='sigmoid', name = 'covtr14', kernel_initializer=init, padding = 'valid' )(x)

generator = keras.Model(inputs = gen_inp, outputs = gen_out, name = 'Generator' )
generator.summary()


# DISCRIMINATOR
disc_inp = Input(shape = input_shape, name = 'disc_inp')
x = Conv2D(128, (4, 4), strides = (2, 2), name = 'firstDiscLayer', kernel_initializer=init)(disc_inp)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), strides = (2, 2) , name='secondDiscLayer' , kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), strides = (2, 2), name = 'thirdDiscLayer', kernel_initializer=init)(x)
x = LeakyReLU(alpha = 0.2)(x)
x = BatchNormalization()(x)

x = tf.keras.layers.Flatten()(x)
disc_out = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer=init)(x)

discriminator = keras.Model(inputs = disc_inp, outputs = disc_out, name = 'Discriminator')
discriminator.summary()


@tf.function
def step_train(data, batch_size, train_real=True):
    """training loop logic for gan. train_real used to train the discriminator
    on either batches of real or fake data
    """

    d_loss = 0.0
    g_loss = 0.0
    data = tf.cast(data, tf.float16)
    a, b, c = input_shape
    data = tf.reshape(data, (-1, a, b, c))
    
    # If training with both real and generated data in the same batch and then
    # compute the loss similarly as below without train_real
    # images = tf.concat([generated_images, data], axis=0)
    # labels = tf.concat(
    #     [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis = 0)
    
    # Discriminator
    with tf.GradientTape() as disc_tape:
        random_latent = tf.random.normal((batch_size,) + latent_dim)
        
        # if and else used so that the discriminiator either only discriminates
        # either real or fake images in one batch
        if train_real:
            real_images = discriminator(data, training=True)
            labels = tf.ones_like(real_images)
            d_loss = discriminator_loss(labels, real_images)
            disc_grad = disc_tape.gradient(
                d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(
                zip(disc_grad, discriminator.trainable_variables))

        else:
            generated_images = generator(random_latent, training=True)
            fake_images = discriminator(generated_images, training=True)
            labels = tf.zeros_like(fake_images)
            d_loss = discriminator_loss(labels, fake_images)
            disc_grad = disc_tape.gradient(
                d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(
                zip(disc_grad, discriminator.trainable_variables))
            
    
    
    # Generator
    with tf.GradientTape() as gen_tape:
        generated_images = generator(random_latent, training=True)
        fake_output = discriminator(generated_images, training=True)
        g_loss = generator_loss(fake_output)
        gen_grad = gen_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(
        zip(gen_grad, generator.trainable_variables))
    
    return g_loss, d_loss

@tf.function
def discriminator_loss(labels, images):
    """returns loss value of discriminator for given data
    """
    labels = tf.multiply(0.9, labels)
    return loss_func(labels, images)

@tf.function
def generator_loss(fake_output):
    """returns loss value of generator for given data
    """
    
    labels = tf.ones_like(fake_output)
    return loss_func(labels, fake_output)
try:
    os.mkdir(f'./Models/gans/{save_modelandimages}/')
except:
    pass

try:
    os.mkdir(f'./ganimages/{save_modelandimages}Epochs/')
except:
    pass
def start_train(dataset, epochs=3, batch_size=10):
    """train the gan model on the given dataset for a specified number of 
    epochs and batchsize.
    """
    global save_modelandimages
    split_factor = dataset.data.shape[0]/batch_size
    for epoch in range(epochs):
        print("Epoch Number", epoch + 1, end=' ')
        d_losses = []
        g_losses = []
        
        
        # Splitting epochs into odd and even ensures that while training the 
        # discriminator all available data is used
        if epoch%2==0:
            for step, real_images in enumerate(np.split(dataset.data, split_factor)):
                    if step%2==0:
                        losses = step_train(real_images, batch_size)
                        d_losses.append(losses[1])
                        g_losses.append(losses[0])
                    else:
                        losses =  step_train(real_images, batch_size, False)
                        d_losses.append(losses[1])
                        g_losses.append(losses[0])
        else:
            for step, real_images in enumerate(np.split(dataset.data, split_factor)):
                    if step%2==0:
                        losses = step_train(real_images, batch_size, False)
                        d_losses.append(losses[1])
                        g_losses.append(losses[0])
                    else:
                        losses =  step_train(real_images, batch_size)
                        d_losses.append(losses[1])
                        g_losses.append(losses[0])
                
        print('d_loss:', np.mean(d_losses), 'g_loss:', np.mean(g_losses))
        image = generator.predict(tf.random.normal((1, latent_dim[0]), seed = 1))
        a, b, c = input_shape
        image = image.reshape(a, b, c)
        image = np.clip(image, 0, 1)
        try:
            os.mkdir(f'./Models/gans/{save_modelandimages}/{epoch}')
        except:
            pass
        plt.imsave(
            f'./ganimages/{save_modelandimages}Epochs/{epoch}.jpg', image)
        
        tf.keras.models.save_model(
            generator, f'./Models/gans/{save_modelandimages}/{epoch}')
        del d_losses, g_losses

def show_images(images, cols=1, titles=None):
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
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
start_train(dataset.data, epochs = 200, batch_size = 25)