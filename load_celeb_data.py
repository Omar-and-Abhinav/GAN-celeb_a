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
        
    def load_data(self, number_of_photos = 'max', ):
        os.chdir(f'.\\{self.folder_name}')
        
        if number_of_photos == 'max':
            number_of_photos = len(glob.glob('*'))
        
        for filename in glob.glob('*')[:number_of_photos]:
            image = plt.imread(filename)
            self.data.append(image)
            
        
        os.chdir('..')
            
    def show_image(self, image_number):
        try:
            assert len(self.data) != 0
            image_number -= 1
            plt.imshow(self.data[image_number])
        
        except AssertionError:
            print('Data not loaded')
    
    def grayscale(self):
        # ToDo: take the loaded data and convert to greyscale
        pass
    
    def color_channels(self):
        # ToDo: Return a list of 3 lists, each list representing either R,G or 
        # B values
        pass
    

            

