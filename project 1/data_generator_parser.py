# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:14:25 2024

@author: Mattia
"""
import configparser
import json
from const import IMG_MIN_DIM, IMG_MAX_DIM

class DataGeneratorConfig:
    def get_config(self, path_name='data_generator.ini'):
        """
        Read the configuration file initialize the configuration object DataGeneratorConfig

        Parameters
        ----------
        path_name : String, optional
            path of the configuration file. The default is 'data_generator.ini'.

        """
        #read .ini file to get all the parameters for the following custom run
        config = configparser.ConfigParser()
        
        # read file
        config.read(path_name) #'data_generator.ini'
        
        # get values
        self.img_dimension = config.getint('data_feature','img_dimension')
        self.noise_fraction = config.getfloat('data_feature','noise_fraction')
        self.rectangle_height = json.loads(config.get('data_feature','rectangle_height'))
        self.rectangle_width = json.loads(config.get('data_feature','rectangle_width'))
        self.vertical_bar_width = json.loads(config.get('data_feature','vertical_bar_width'))
        self.horizontal_bar_width = json.loads(config.get('data_feature','horizontal_bar_width'))
        self.circle_radius = json.loads(config.get('data_feature','circle_radius'))
        self.cross_dimension = json.loads(config.get('data_feature','cross_dimension'))
        self.flat = config.getboolean('data_feature','flat')
        self.center = config.getboolean('data_feature','center')
        
        self.train_size = config.getfloat('dataset','train_size')
        self.validation_size = config.getfloat('dataset','validation_size')
        self.test_size = config.getfloat('dataset','test_size')
        self.num_images_for_class = config.getint('dataset','num_images_for_class')

        if(self.img_dimension < IMG_MIN_DIM or self.img_dimension > IMG_MAX_DIM):  
            raise ValueError("Invalid img dimension")
            
    def __str__(self):
        #print object
        
        feats = vars(self)
        feats_formatted = ', '.join(f"{k}={v}" for k, v in feats.items())
        return f"{self.__class__.__name__}({feats_formatted})"

if __name__ == "__main__":
    data_config = DataGeneratorConfig()
    data_config.get_config()
    print(data_config)
    #print(type(data_config.cross_dimension[0]))