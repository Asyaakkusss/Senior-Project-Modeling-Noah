# Imports
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
#from sklearn.preprocessing import LabelEncoder

# Start off with Asya's my_kalman_filter.py steps: load, convert from array to integer, then process the data

# Load in data from PureSleepTime.csv, HKCategoryTypeIdentifierSleepAnalysis.csv, BasalMetabolicRate.csv
with open(r'../data/HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
# with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_analysis = np.array(column_data) 
with open(r'../data/PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_time = np.array(column_data)
with open(r'../data/PureSleepTime.csv', 'r') as file:
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


# =============================================================================================
# ============================= DATA PROCESSING ===============================================
# =============================================================================================

# data processing for sleep time
df_st = pd.read_csv("../data/PureSleepTime.csv")
#df_st = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/data/PureSleepTime.csv")
#df_st = pd.read_csv("F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv")

# convert to datetime 
df_st['start'] = pd.to_datetime(df_st['time'])

#the datetime values will be used 
df_st.set_index('start', inplace=True)

if df_st.index.duplicated().any():
    df_st1 = df_st[~df_st.index.duplicated(keep='first')]

start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
end_time = pd.Timestamp('2024-09-05 08:27:27-0400')

#normalize them to a constant frequency 
common_time = pd.date_range(start=start_time, 
                            end=end_time, 
                            freq='min')

#align values with the times 
sleep_time_interpolated = df_st['value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_sleep_time_df = pd.DataFrame({
    'value': sleep_time_interpolated
})

print(aligned_sleep_time_df)

all_nan_sleep_time = aligned_sleep_time_df.dropna()
processed_sleep_time = all_nan_sleep_time.to_numpy().flatten()
sleep_time_sans_nan = processed_sleep_time[~all_nan_sleep_time]

print(sleep_time_sans_nan)

# ============================================================================================================================
# ============================================================================================================================

# data processing for sleep analysis 
df_sa_original = pd.read_csv('..data/HKCategoryTypeIdentifierSleepAnalysis.csv')
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
df_sa_original['onehot_encoded_value'] = df_sa_original['value'].map(category_mapping)
df_sa_original.to_csv('../data/SleepAnalysis_label_data.csv', index=False)
#df_sa_original.to_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv', index=False)  # save to new csv file
print(df_sa_original)

# Now we can work with the date time 

# Read the new processed csv file --- change the names for df here to represent sleep analysis -> df_sa
df_sa = pd.read_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv')
# convert to datetime 
df_sa['start'] = pd.to_datetime(df_sa['start'])

#the datetime values will be used 
df_sa.set_index('start', inplace=True)

if df_sa.index.duplicated().any():
    df_sa = df_sa[~df_sa.index.duplicated(keep='first')]

start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
#normalize them to a constant frequency 
common_time = pd.date_range(start=start_time, 
                            end=end_time, 
                            freq='min')

#align values with the times 
sleep_analysis_interpolated = df_sa['onehot_encoded_value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_sleep_analysis_df = pd.DataFrame({
    'onehot_encoded_value': sleep_analysis_interpolated
})

processed_sleep_analysis = aligned_sleep_analysis_df.to_numpy().flatten()
all_nan_sleep_analysis = np.isnan(processed_sleep_analysis)
sleep_analysis_sans_nan = processed_sleep_analysis[~all_nan_sleep_analysis]
# print(processed_sleep_analysis)

# ====================================================================================================================

# data processing for basal metabolic rate 
df_be = pd.read_csv("F:\FALL 2024\Senior-Project-Modeling-Noah\data\BasalEnergyBurned.csv")

# convert to datetime 
df_be['start'] = pd.to_datetime(df_be['start'])

#the datetime values will be used 
df_be.set_index('start', inplace=True)

if df_be.index.duplicated().any():
    df_be = df_be[~df_be.index.duplicated(keep='first')]

start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
#normalize them to a constant frequency 
common_time = pd.date_range(start=start_time, 
                            end=end_time, 
                            freq='min')

#align values with the times 
basal_interpolated = df_be['value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_basal_df = pd.DataFrame({
    'value': basal_interpolated
})

processed_basal_rate = aligned_basal_df.to_numpy().flatten()
all_nan_basal_rate = np.isnan(processed_basal_rate)
basal_rate_sans_nan = processed_basal_rate[~all_nan_basal_rate]

# =============================================================================================================

# Currently, becuase the different data sets have different number of nans, the data sizes are different, 
# So we use the minimum length of all the data and use that size for all the arrays of data

# Ensuring consistent length of the arrays
min_length = min(len(basal_rate_sans_nan), len(sleep_analysis_sans_nan), len(sleep_time_sans_nan))
basal_rate_sans_nan = basal_rate_sans_nan[:min_length]
sleep_analysis_sans_nan = sleep_analysis_sans_nan[:min_length]
sleep_time_sans_nan = sleep_time_sans_nan[:min_length]

# Convert to 1D arrays
basal_rate_sans_nan = np.array(basal_rate_sans_nan).flatten()
sleep_analysis_sans_nan = np.array(sleep_analysis_sans_nan).flatten()
sleep_time_sans_nan = np.array(sleep_time_sans_nan).flatten()

# =================================================================================================================
# =================================================================================================================
# ================================================    PLOTTING GRAPH     ==========================================
# =================================================================================================================
# =================================================================================================================
