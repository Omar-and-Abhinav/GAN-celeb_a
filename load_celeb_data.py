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
        self.data_gray=[]
        self.data_color=[]
        
    
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
            plt.imshow(self.data[image_number])   #shape=(109,89,3) for each image
    
            
        except AssertionError:
            print('Data not loaded')
    
    
    def grayscale(self,image_number=0):
        for img in self.data:
            r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]   
            gray=0.2989*r + 0.5870*g + 0.1140*b      
            self.data_gray.append(gray)
            
        
        plt.imshow(self.data_gray[image_number],cmap=plt.get_cmap('gray'))
        
    
    def color_channels(self,red=0,green=0,blue=0,image_number=0):
        redPixs=[]
        greenPixs=[]
        bluePixs=[]
        diffPixs=[]
        for img in self.data:
            r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
            redPixs.append(r)
            greenPixs.append(g)
            bluePixs.append(b)
            if red==1:
                img=r+ g*0 +b*0
                self.data_color.append(img)
                colour='Reds'
            elif blue==1:
                img=r*0+ g*0 +b
                self.data_color.append(img)
                colour='Blues'
            elif green==1:
                img=r*0+ g +b*0
                self.data_color.append(img)
                colour='Greens'
        
        diffPixs.append(redPixs)
        diffPixs.append(greenPixs)
        diffPixs.append(bluePixs)
        
        plt.imshow(self.data_color[image_number],cmap=plt.get_cmap(colour))
        print("len of redPixels for one image:",redPixs[0].shape)
        #print("RED PIXELS\n",redPixs,'\n',"GREEN PIXELS:\n",greenPixs,"BLUE PIXELS:\n",bluePixs)
        
        return diffPixs
        
       
     
        
        
    

one_data=photo_dataset()
one_data.load_data(4)
different_pixels=one_data.color_channels(blue=1,image_number=2)
#print("shape of each image:",one_data.data[2].shape)


