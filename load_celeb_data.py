import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class photo_dataset():
    
    '''A class that provides added functionality for a
    a dataset of photos in a folder.'''
    
    
    def __init__(self, folder_name = 'celeb_a'):
        self.folder_name = folder_name
        self.data = []
        self.data_gray = []
        self.data_color = {"red":[], "green":[], "blue":[]}
        
    
    def load_data(self, number_of_photos = 'max', ):
        os.chdir(f'.\\{self.folder_name}')
        
        if number_of_photos == 'max':
            number_of_photos = len(glob.glob('*'))
           
        
        for filename in glob.glob('*')[:number_of_photos]:
            image = plt.imread(filename)
            self.data.append(image)
        self.data = np.array(self.data)
        
        
        os.chdir('..')
            
    
    def show_image(self, image_number):
        try:
            assert len(self.data) != 0
            #shape=(109,89,3) for each image
            plt.imshow(self.data[image_number])   
    
            
        except AssertionError:
            print('Data not loaded')
    
    
    def grayscale(self, image_number=None):
        for img in self.data:
            r, g, b = img[:, :, 0], img[:, :, 1],img[:, :, 2]   
            gray = 0.2989*r + 0.5870*g + 0.1140*b      
            self.data_gray.append(gray)
            
        if image_number != None:
            plt.imshow(self.data_gray[image_number],cmap=plt.get_cmap('gray'))
        
    
    def color_channels(self, red=0, green=0, blue=0):
        redPixs = []
        greenPixs = []
        bluePixs = []
        for img in self.data:
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
    
    def noisy_data(self, number_of_photos = 1, mean=0, std=1, by255=False):
        noisy_data = []
        factor = 1
        if by255:
            factor = 255
        for image in self.data[:number_of_photos]:
            # Add Guassian Noise to each image
            noisy_image = image + np.random.normal(mean, std, image.shape)
            # Keep the values between 0 and 255
            noisy_image = np.clip(noisy_image, 0, 255) 
            noisy_data.append(noisy_image/factor)
        return noisy_data