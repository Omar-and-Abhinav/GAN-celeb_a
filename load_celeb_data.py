import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class photo_dataset():
    '''A class that provides added functionality for a
    a dataset of photos in a folder. Requires the name of the folder where the
    data is stored(should be a subdirectory.)
    '''

    def __init__(self, folder_name='celeb_a'):
        self.folder_name = folder_name
        self.data = []
        self.data_color = {"red": [], "green": [], "blue": []}

    def load_data(self, number_of_photos='max', scale = 0):
        '''Load a dataset of photos. The dataset should be stored as a 
        subfolder of the directory where this python file is located. By
        default all the photos are loaded, but a limit can be specified 
        through the number_of_photos parameter.
        '''

        os.chdir(f'.\{self.folder_name}')
        filenames = glob.glob('*')
        if number_of_photos == 'max':

            number_of_photos = len(filenames)

        for filename in filenames[:number_of_photos]:
            image = Image.open(filename)
            if scale == 1:
                self.data.append(np.array(image, dtype = np.float32)/255)
            else:
                self.data.append((np.array(image, dtype = np.float32)))
            image.close()
        self.data = np.array(self.data, dtype = np.float32)

        os.chdir('..')

    def show_image(self, image_number):
        '''Shows an image from the loaded dataset using matplotlib. Starts 
        from 0.
        '''

        try:
            assert len(self.data) != 0
            # shape=(109,89,3) for each image
            plt.imshow(self.data[image_number])

        except AssertionError:
            print('Data not loaded')

    def grayscale(self, images, image_number=None):
        '''Stores a grayscale version of the passed data in data_grey
        '''
        data_gray = []
        for img in images:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989*r + 0.5870*g + 0.1140*b
            data_gray.append(np.array(gray, dtype = np.float32))
            del gray, r, g, b
        
        if image_number != None:
            plt.imshow(data_gray[image_number], cmap=plt.get_cmap('gray'))
        
        return np.array(data_gray, dtype = np.float32)
    def color_channels(self, number_of_photos=1, red=0, green=0, blue=0):
        '''Stores the respective red, green and blue channels of the photos
        in data_color dictionary with keys red, green, blue.
        '''
        redPixs = []
        greenPixs = []
        bluePixs = []
        for img in self.data[:number_of_photos]:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            redPixs.append(np.array(r))
            greenPixs.append(np.array(g))
            bluePixs.append(np.array(b))
        if red == 1:
            self.data_color["red"] = np.array(redPixs)

        if blue == 1:
            self.data_color["blue"] = np.array(greenPixs)

        if green == 1:
            self.data_color["green"] = np.array(bluePixs)

    def noisy_data(self, list_images, mean=0, std=1, by255=False):
        '''Returns a subset of the dataset with Gaussian Noise. Set by255 to 
        True for values between 0 and 1.
        '''
        noisy_data = []
        factor = 1
        if by255:
            factor = 255
        for image in list_images:
            # Add Guassian Noise to each image
            noisy_image = image + np.random.normal(mean, std, image.shape)
            # Keep the values between 0 and 255
            noisy_image = np.clip(noisy_image, 0, 255)
            noisy_data.append(noisy_image/factor)
        return np.array(noisy_data)

    def split_data(self, list_images, trainperc, valperc):
        '''splits the data into testing and training data'''

        # number of images in training set
        length = round(trainperc * len(list_images))
        # number of images validation
        ratio = round(len(list_images) * valperc)

        xTrain = np.array(list_images[:length])/255
        xVal = np.array(list_images[length: (length + ratio)])/255
        xTest = np.array(list_images[(length + ratio):])/255

        print('    SPLITTED THE DATA INTO TRAINING AND TEST SET   ')
        print('Number of Training samples:', len(xTrain))
        print('Number of Validation samples:', len(xVal))
        print('Number of Test samples:', len(xTest))
        
        

        return xTrain, xVal, xTest