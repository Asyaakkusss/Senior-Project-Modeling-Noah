# Imports
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import LabelEncoder

#both of inputs need to be strings. column of interest is time for pure sleep time and start for the basal metabolic rate and the hk sleep time
def process_numerical_data(csv_string, column_of_interest): 
    # data processing for sleep time
    dataframe = pd.read_csv(csv_string)

    # convert to datetime 
    dataframe['start'] = pd.to_datetime(dataframe[column_of_interest])

    #the datetime values will be used 
    dataframe.set_index('start', inplace=True)

    if dataframe.index.duplicated().any():
        dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

    start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
    end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
    #normalize them to a constant frequency 
    common_time = pd.date_range(start=start_time, 
                                end=end_time, 
                                freq='min')

    #align values with the times 
    interpolated_data = dataframe['value'].reindex(common_time).interpolate()

    #create a dataframe with start and value columns 
    aligned_dataframe = pd.DataFrame({
        'value': interpolated_data
        })

    processed_data = aligned_dataframe.to_numpy().flatten()
    all_nan_data = np.isnan(processed_data)
    non_nan_processed_data = processed_data[~all_nan_data]

    return non_nan_processed_data

#column of interest is "onehot_encoded_value", mapped column is "value". feed the generated csv into the
#process_numerical_data method after running this method. 
def process_categorical_data(csv_string, column_of_interest, mapped_column, category_mapping): 

    # data processing for sleep analysis 
    df_sa_original = pd.read_csv(csv_string)

    # First we are adding the column with the one hot encoded value to 'quantify' the sleep analysis data

    
    # Map categories to numeric valuesby one hot encoding
    df_sa_original[column_of_interest] = df_sa_original[mapped_column].map(category_mapping)
    df_sa_original.to_csv('data/HKCategoryTypeIdentifierSleepAnalysis_processed.csv', index=False)  # save to new csv file

def calc_R(arrays):     
    unified_array = np.array(arrays)
    R = np.cov(unified_array)
    return R 