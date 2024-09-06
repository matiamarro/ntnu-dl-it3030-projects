# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:03:34 2024

@author: Mattia
"""
import pandas as pd

def feature_engenering(dataframe):

    df = dataframe.copy()
    
    #converting timestamp to time object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    #time features
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['hour'] = df['timestamp'].dt.hour
    
    #shifting temperature to do a temperature forecast
    num_positions = 24
    df['NO1_temperature'] = df['NO1_temperature'].shift(periods=num_positions, fill_value=0)
    df['NO2_temperature'] = df['NO2_temperature'].shift(periods=num_positions, fill_value=0)
    df['NO3_temperature'] = df['NO3_temperature'].shift(periods=num_positions, fill_value=0)
    df['NO4_temperature'] = df['NO4_temperature'].shift(periods=num_positions, fill_value=0)
    df['NO5_temperature'] = df['NO5_temperature'].shift(periods=num_positions, fill_value=0)
    
    df.set_index('timestamp', inplace=True)
    
    return df