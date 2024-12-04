# Imports
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
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

    # Map categories to numeric valuesby one hot encoding
    df_sa_original['value'] = df_sa_original['value'].map(category_mapping)
    
    #I don't think it's helpful to save the csv...
    df_sa_original.to_csv('data/SleepAnalysis_label_data.csv', index=False)
    #df_sa_original.to_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv', index=False)  # save to new csv file

    # Now we can work with the date time 
    df_sa_filtered = df_sa_original[df_sa_original['value'] != 2]
    # Read the new processed csv file --- change the names for df here to represent sleep analysis -> df_sa
    #df_sa = pd.read_csv('data/SleepAnalysis_label_data.csv')
    #df_sa = pd.read_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv')
    
    # convert to fit ranges
    df_sa = adapt_change(df_sa_filtered, "Sleep_Analysis")
    
    return df_sa

def adapt_change(time_df: pd.DataFrame, data_name) -> pd.DataFrame:
    #the datetime values will be used 
    time_df.set_index('start', inplace=True)

    if time_df.index.duplicated().any():
        time_df = time_df[~time_df.index.duplicated(keep='first')]

    start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
    end_time = pd.Timestamp('2024-09-05 08:27:27-0400')

    #normalize them to a constant frequency 
    common_time = pd.date_range(start=start_time, 
                                end=end_time, 
                                freq='min')

    #align values with the times 
    print("Original time_df:\n", time_df.head())
    #interpolated_df: pd.DataFrame = time_df['value'].reindex(common_time).interpolate()
    #print("After reindex and interpolate:\n", interpolated_df.head())

    # Ensure both are sorted before merging
    time_df = time_df.sort_index()
    common_time_df = pd.DataFrame(index=common_time)

    aligned_df = pd.merge_asof(
        common_time_df, time_df, left_index=True, right_index=True, direction='nearest'
    )

    aligned_df = pd.DataFrame({
    'value': time_df['value']
    }, index=time_df.index)

    processed_df = aligned_df.dropna()
    #df_sans_nan = processed_df.to_numpy().flatten()

    print("Length of common_time:", len(common_time))
    print("Length of processed analysis:", len(processed_df))
    print(processed_df)
    return processed_df

def convert_1D():
    sleep_time_sans_nan = regularize_time()
    sleep_analysis_sans_nan = regularize_analysis()
    basal_rate_sans_nan = regularize_metabolism()

    # Currently, becuase the different data sets have different number of nans, the data sizes are different, 
    # So we use the minimum length of all the data and use that size for all the arrays of data

    # Ensuring consistent length of the arrays
    min_length = min(len(basal_rate_sans_nan), len(sleep_analysis_sans_nan), len(sleep_time_sans_nan))
    basal_rate_sans_nan = basal_rate_sans_nan[:min_length]
    sleep_analysis_sans_nan = sleep_analysis_sans_nan[:min_length]
    sleep_time_sans_nan = sleep_time_sans_nan[:min_length]

    # Convert to 1D arrays
    basal_rate_sans_nan_1D = np.array(basal_rate_sans_nan).flatten()
    sleep_analysis_sans_nan_1D = np.array(sleep_analysis_sans_nan).flatten()
    sleep_time_sans_nan_1D = np.array(sleep_time_sans_nan).flatten()

    return basal_rate_sans_nan_1D, sleep_analysis_sans_nan_1D, sleep_time_sans_nan_1D

