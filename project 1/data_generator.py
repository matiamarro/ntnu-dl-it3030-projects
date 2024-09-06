# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:20:46 2024

@author: Mattia
"""
from enum import Enum 
from PIL import Image
import random
from random import randint
import numpy as np
import os
from data_generator_parser import DataGeneratorConfig
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder

class Classes(Enum):
    RECTANGLE = 0
    HORIZONTALBAR = 1
    CIRCLE = 2
    VERTICALBAR = 3

class ImgGenerator:
    def __init__(self, param):
        self.param = param
        self.img_folder = "imgs"
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
    
    def print_samples(self):
        """
        plot 10 images
        """
        imgs,_ = self.get_images()
        
        #reshape in case imgs are flat
        imgs = [img.reshape((self.param.img_dimension,self.param.img_dimension)) for img in imgs]
        
        num_samples = min(len(imgs), 10)
        
        #random samples selection 
        random_indices = random.sample(range(len(imgs)), num_samples)
        
        selected_samples = [imgs[i] for i in random_indices]
        
        for binary_matrix in selected_samples:
            int_matrix = np.array(binary_matrix, dtype=int)
        
            # Plot the image using imshow
            plt.imshow(int_matrix, cmap='gray', aspect='auto')
            plt.axis('off')
            plt.show()
    
    def get_images(self, flat=False):
        """
        
        Returns images already created
        ----------
        data: nparray (num_sample, matrx/vector img dimension)
        label: (num_samples, [one_hot_encode_class])
        
        """
        data = []
        label = []
                    
        for file_name in os.listdir(self.img_folder):
            # get path file
            file_path = os.path.join(self.img_folder, file_name)
    
            # Vverify if the file is an image (.png)
            if file_name.lower().endswith('.png'):
                # load image
                img = Image.open(file_path)
    
                # image to array numpy
                array_img = np.array(img)

                if(self.param.flat == True):
                    array_img = array_img.flatten()
                
                array_img_bool = array_img != 0
    
                # add array_image into the list
                if flat == True:
                    data.append(array_img_bool.astype(int).flatten())
                else:
                    data.append(array_img_bool.astype(int))
                
                class_name = file_name.split("_")[0]
                label.append(Classes[class_name].value)
        
        data = np.array(data)
        label = np.array(label)
        
        # Create an instance of OneHotEncoder for the labels
        encoder = OneHotEncoder(sparse=False)
        
        # Reshape the vector to a column vector as fit_transform expects 2D input
        vector_reshaped = label.reshape(-1, 1)
        
        # Fit and transform the data
        label = encoder.fit_transform(vector_reshaped)
        
        return data, label
    
    def create_images(self):
        """
        Create the images saving them on file system as png

        """
        #rectangle
        rectangles = self.__create_rectangle()
        #hor_bar
        horizontal_bars = self.__create_horizontal_bars()
        #circle
        circles = self.__create_circles()
        #vert_bars
        vertical_bars = self.__create_vertical_bars()
        
        #add noise
        rectangles = [self.__add_noise(rectangle) for rectangle in rectangles]
        horizontal_bars = [self.__add_noise(bar) for bar in horizontal_bars]
        circles = [self.__add_noise(circle) for circle in circles]
        vertical_bars = [self.__add_noise(bar) for bar in vertical_bars]
        
        def save_imgs(imgs_vector, name):
            for index, img_array in enumerate(imgs_vector):
                # from bool to int array 
                image = np.uint8(img_array) * 255
            
                #from array to Image object
                image = Image.fromarray(image)
            
                #saving the image using the name of his class inside
                name_file = f"{self.img_folder}/{name}_{index + 1}.png"
                image.save(name_file)
        
        #save img for 4 classes
        save_imgs(rectangles, Classes.RECTANGLE.name)
        save_imgs(horizontal_bars, Classes.HORIZONTALBAR.name)
        save_imgs(circles, Classes.CIRCLE.name)
        save_imgs(vertical_bars, Classes.VERTICALBAR.name)
    
    def __add_noise(self, img):
        
        total_pixels = img.size
        pixels_to_flip = int(self.param.noise_fraction * total_pixels)
        
        # Get random indices to flip pixels
        indices_to_flip = np.random.choice(total_pixels, pixels_to_flip, replace=False)
        
        # Flip selected pixels (change 0 to 1 and vice versa)
        image_array_flipped = img.copy()
        image_array_flipped.flat[indices_to_flip] = 1 - image_array_flipped.flat[indices_to_flip]
        
        # Reshape back to the original shape
        image_array_flipped = image_array_flipped.reshape(img.shape)
        
        return image_array_flipped
        
    def __create_rectangle(self):
        rectangles = []
        rectanlge_range_height = self.param.rectangle_height
        rectanlge_range_width = self.param.rectangle_width
        
        for i in range(self.param.num_images_for_class):
            if(self.param.center):
                center_x = (self.param.img_dimension // 2)
                center_y = center_x
                if(self.param.img_dimension % 2 == 0):
                    max_half_height = self.param.img_dimension / 2 - 1
                    max_half_width = max_half_height
                else:
                    max_half_height = self.param.img_dimension // 2
                    max_half_width = max_half_height

                max_half_height = min(
                    max_half_height, rectanlge_range_height[1] // 2)
                max_half_width = min(
                    max_half_width, rectanlge_range_width[1] // 2)

                height = randint(
                    rectanlge_range_height[0] // 2, max_half_height)
                width = randint(rectanlge_range_width[0] // 2, max_half_width)

                start_y = center_y - height
                end_y = center_y + height
                start_x = center_x - width
                end_x = center_x + width
            else:
                start_y = randint(0, self.param.img_dimension -
                                  rectanlge_range_height[0] - 1)
                start_x = randint(0, self.param.img_dimension -
                                  rectanlge_range_width[0] - 1)

                max_y = min(self.param.img_dimension - 1,
                            start_y + rectanlge_range_height[1])
                max_x = min(self.param.img_dimension - 1,
                            start_x + rectanlge_range_width[1])

                end_y = randint(start_y + rectanlge_range_height[0], max_y)
                end_x = randint(start_x + rectanlge_range_width[0], max_x)

            img = np.zeros((self.param.img_dimension,
                            self.param.img_dimension))

            img[start_y:end_y, start_x] = 1
            img[start_y:end_y, end_x] = 1
            img[start_y, start_x:end_x] = 1
            img[end_y, start_x:end_x] = 1

            img[end_y, end_x] = 1 

            rectangles.append(img)

        return rectangles
    
    def __create_horizontal_bars(self):
       horizontal_bars = []
       horizontal_bar_width = self.param.horizontal_bar_width
       
       for i in range(self.param.num_images_for_class):
           this_bar_width = randint(
               horizontal_bar_width[0], horizontal_bar_width[1])

           indexes = []
           k = 0
           for j in range(self.param.img_dimension):
               if k < this_bar_width:
                   indexes.append(j)

               k += 1
               if k >= this_bar_width * 4:
                   k = 0

           img = np.zeros((self.param.img_dimension,
                           self.param.img_dimension))

           img[indexes] = 1
           img = np.roll(img, randint(0, self.param.img_dimension), axis=0)

           horizontal_bars.append(img)

       return horizontal_bars

    def __create_circles(self):
        circles = []
        circle_radius_range = self.param.circle_radius
        
        for i in range(self.param.num_images_for_class):
            if(self.param.center):
                center_x = (self.param.img_dimension // 2)
                center_y = center_x
                if(self.param.img_dimension % 2 == 0):
                    max_radius = self.param.img_dimension / 2 - 1
                else:
                    max_radius = self.param.img_dimension // 2
                min_radius = circle_radius_range[0]
                max_radius = min(max_radius, circle_radius_range[1])
                radius = randint(min_radius, max_radius)
            else:
                center_x = randint(
                    circle_radius_range[0], self.param.img_dimension - circle_radius_range[0] - 1)
                center_y = randint(
                    circle_radius_range[0], self.param.img_dimension - circle_radius_range[0] - 1)
                max_radius = min(self.param.img_dimension - center_x - 1,
                                 center_x, self.param.img_dimension - center_y - 1, center_y)
                max_radius = min(max_radius, circle_radius_range[1])
                radius = randint(circle_radius_range[0], max_radius)

            img = np.zeros((self.param.img_dimension,
                            self.param.img_dimension))

            for degree in range(0, 360, 5):
                x = round(radius * math.cos(math.radians(degree)))
                y = round(radius * math.sin(math.radians(degree)))
                img[center_y + y, center_x + x] = 1

            circles.append(img)

        return circles
    
    def __create_vertical_bars(self):
       vertical_bars = []
       vertical_bar_width = self.param.vertical_bar_width
       
       for i in range(self.param.num_images_for_class):
           this_bar_width = randint(
               vertical_bar_width[0], vertical_bar_width[1])

           indexes = []
           k = 0
           for j in range(self.param.img_dimension):
               if k < this_bar_width:
                   indexes.append(j)

               k += 1
               if k >= this_bar_width * 4:
                   k = 0

           img = np.zeros((self.param.img_dimension,
                           self.param.img_dimension))

           img[indexes] = 1
           img = np.roll(img, randint(0, self.param.img_dimension), axis=0)

           vertical_bars.append(img)
           
       vertical_bars = [np.transpose(bar) for bar in vertical_bars]    

       return vertical_bars
        
if __name__ == "__main__":
    data_config = DataGeneratorConfig()
    data_config.get_config()
    #print(data_config)
    img_gen = ImgGenerator(data_config)
    img_gen.create_images()
    d, l = img_gen.get_imgs()
    img_gen.print_samples()
    
        