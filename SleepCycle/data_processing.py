# Imports
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
import graph_test
from datetime import timedelta

#from sklearn.preprocessing import LabelEncoder

# Start off with Asya's my_kalman_filter.py steps: load, convert from array to integer, then process the data

# This does nothing
'''
# Load in data from PureSleepTime.csv, HKCategoryTypeIdentifierSleepAnalysis.csv, BasalMetabolicRate.csv
with open(r'data/HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
# with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_analysis = np.array(column_data) 
with open(r'data/PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_time = np.array(column_data)
with open(r'data/PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\BasalEnergyBurned.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\BasalEnergyBurned.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

basal_energy = np.array(column_data)

#convert the arrays to integers 
def convert_to_integer(array): 
    return np.array(array).astype(float)

ST = convert_to_integer(sleep_time)
BE = convert_to_integer(basal_energy)
'''

# =============================================================================================
# ============================= DATA PROCESSING ===============================================
# =============================================================================================


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
    
    time_series = np.array(interpolated_data.index)
    
    #create a dataframe with start and value columns 
    aligned_dataframe = pd.DataFrame({
        'value': interpolated_data
        })

    processed_data = aligned_dataframe.to_numpy().flatten()
    print(aligned_dataframe.columns)
    all_nan_data = np.isnan(processed_data)
    non_nan_processed_data = processed_data[~all_nan_data]
    time_series_present = time_series[~all_nan_data]
    #present_time_data = processed_time[]

    return non_nan_processed_data,time_series_present

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

def calc_X(hidden_var_estimate, arrays): 
    # Calculate the mean for each array after slicing
    means = [np.mean(arr[slice_start:]) for arr in arrays]
    
    # Combine the hidden_var_estimate with the means
    X = np.array([hidden_var_estimate] + means).reshape(-1, 1)
    
    return X

def regularize_time():
    df_st_original = pd.read_csv("data/PureSleepTime.csv")
    df_st_original['start'] = pd.to_datetime(df_st_original['time'])
    df_st_original.to_csv('data/SleepTime_label_data.csv', index=False)
    #df_st = pd.read_csv("F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv")

    df_st = adapt_change(df_st_original, "Sleep_Time")

    return df_st

def regularize_metabolism():
    # data processing for basal metabolic rate 
    df_be_original = pd.read_csv("data/BasalEnergyBurned.csv")
    df_be_original['start'] = pd.to_datetime(df_be_original['start'])

    df_be = adapt_change(df_be_original, "Basal_Energy_Burned")

    return df_be

def regularize_analysis():
    df_sa_original = pd.read_csv('data/HKCategoryTypeIdentifierSleepAnalysis.csv')
    df_sa_original['start'] = pd.to_datetime(df_sa_original['start'])
    #df_sa_original = pd.read_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv')

    # First we are adding the column with the one hot encoded value to 'quantify' the sleep analysis data

    # Category mapping for the values in the csv
    category_mapping = {
        "HKCategoryValueSleepAnalysisInBed": 2,
        "HKCategoryValueSleepAnalysisAsleepREM": 3,
        "HKCategoryValueSleepAnalysisAsleepDeep": 4,
        "HKCategoryValueSleepAnalysisAsleepCore": 5,
        "HKCategoryValueSleepAnalysisAwake": 1,
        "HKCategoryValueSleepAnalysisAsleepUnspecified": 0,
    }

    
    # Map categories to numeric values
    df_sa_original['value'] = df_sa_original['value'].map(category_mapping)
    #df_sa_filtered = df_sa_original[df_sa_original['value'] != -2]
    
    #I don't think it's helpful to save the csv...
    #df_sa_original.to_csv('data/SleepAnalysis_label_data.csv', index=False)
    #df_sa_original.to_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv', index=False)  # save to new csv file

    # Now we can work with the date time 
    #df_sa_filtered = df_sa_original[df_sa_original['value'] != 2]
    # Read the new processed csv file --- change the names for df here to represent sleep analysis -> df_sa
    #df_sa = pd.read_csv('data/SleepAnalysis_label_data.csv')
    #df_sa = pd.read_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv')
    
    # convert to fit ranges
    df_sa = adapt_change(df_sa_original, "Sleep Analysis")
    
    return df_sa

def adapt_change(time_df: pd.DataFrame, data_name: str) -> pd.DataFrame:
    # Convert 'start' to datetime and set it as the index
    time_df['end'] = pd.to_datetime(time_df['end'], errors='coerce')
    time_df.set_index('end', inplace=True)

    # Drop duplicate indices
    if time_df.index.duplicated().any():
        time_df = time_df[~time_df.index.duplicated(keep='first')]


    # Keep only the 'value' column as a DataFrame
    time_df = time_df[['value']]

    time_df.to_csv('data/SleepAnalysis_label_data.csv')
    print(time_df)

    # Define time range
    start_time = pd.Timestamp('2023-07-18 23:04:52-04:00')
    end_time = pd.Timestamp('2023-07-24 23:11:16-04:00')
    common_time = pd.date_range(start=start_time, end=end_time, freq='s')

    print("Range of time_df.index:", time_df.index.min(), "to", time_df.index.max())
    print("Range of common_time:", common_time.min(), "to", common_time.max())

    #This is to calculate processed_sleep_analysis
    '''
    # Create the new DataFrame
    new_df = pd.DataFrame(index=common_time, columns=['value'])
    filtered_time_df = time_df.loc[start_time:end_time]

    # Extract the old timestamps and values
    old_times = filtered_time_df.index  # Assumes 'end' is set as the index
    old_values = filtered_time_df['value']

    # Initialize iterators
    old_time_index = 0
    
    # Loop through the new DataFrame
    for new_time in new_df.index:
        if old_time_index < len(old_times):
            old_time = old_times[old_time_index]
            old_value = old_values.iloc[old_time_index]

            difference = abs(new_time - old_time)
            print(f"new_time={new_time}, old_time={old_time}, difference={difference}, old_time_index={old_time_index}")
            
            # If the new time closely matches the old time, copy the value
            if difference <= timedelta(seconds=1):
                new_df.loc[new_time] = old_value
                old_time_index += 1  # Move to the next old time
            elif difference >= timedelta(hours=4):
                # Large gap indicates awake state
                new_df.loc[new_time] = 10
            else:
                # Smaller gap, propagate the last known value
                new_df.loc[new_time] = old_value
        else:
            # Fill remaining times with NaN or a default value
            new_df.loc[new_time] = float('nan')


    new_df.to_csv("processed_sleep_analysis.csv")'''
    
    # Read the CSV into a DataFrame
    data = pd.read_csv("processed_sleep_analysis.csv")

    # Convert the 'Unnamed: 0' column to datetime and set it as the index
    data['timestamp'] = pd.to_datetime(data['Unnamed: 0'], errors='coerce')
    data.set_index('timestamp', inplace=True)

    data.drop(columns=['Unnamed: 0'], inplace=True)

    # Resample the data to 1-minute frequency, taking the mean value for each minute
    minute_data = data.resample('T').mean()

    print(f"Regularizing {data_name}")
    print("Length of common_time:", len(common_time))
    print("Length of processed analysis:", len(data))

    minute_data = minute_data.to_csv("minutes_processed_sleep_analysis.csv")

    return minute_data

def convert_1D():

    # Currently, becuase the different data sets have different number of nans, the data sizes are different, 
    # So we use the minimum length of all the data and use that size for all the arrays of data

    #Commenting this out temporarily because we only care about sleep analysis right now
    
    # Ensuring consistent length of the arrays
    min_length = min(len(basal_rate_sans_nan), len(sleep_analysis_sans_nan), len(sleep_time_sans_nan))
    basal_rate_sans_nan = basal_rate_sans_nan[:min_length]
    sleep_analysis_sans_nan = sleep_analysis_sans_nan[:min_length]
    sleep_time_sans_nan = sleep_time_sans_nan[:min_length]

    # Convert to 1D arrays
    basal_rate_sans_nan_1D = np.array(basal_rate_sans_nan).flatten()
    sleep_analysis_sans_nan_1D = np.array(sleep_analysis_sans_nan).flatten()
    sleep_time_sans_nan_1D = np.array(sleep_time_sans_nan).flatten()
    #basal_rate_sans_nan_1D, sleep_analysis_sans_nan_1D, sleep_time_sans_nan_1D

    return sleep_analysis_sans_nan_1D

regularize_analysis()