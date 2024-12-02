# Imports
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import sys
import os

#sys.path.append("F:\FALL 2024\Senior-Project-Modeling-Noah\CoreBodyTemp")
#from my_kalman_filter import n_steps

# Start off with Asya's my_kalman_filter.py steps: load, convert from array to integer, then process the data

# Load in data from PureSleepTime.csv, HKCategoryTypeIdentifierSleepAnalysis.csv, BasalMetabolicRate.csv
with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
# with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_analysis = np.array(column_data) 

with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_time = np.array(column_data)

with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\BasalEnergyBurned.csv', 'r') as file:
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
#df_st = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/data/PureSleepTime.csv")
df_st = pd.read_csv("F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv")

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

processed_sleep_time = aligned_sleep_time_df.to_numpy().flatten()
all_nan_sleep_time = np.isnan(processed_sleep_time)
sleep_time_sans_nan = processed_sleep_time[~all_nan_sleep_time]

# ============================================================================================================================
# ============================================================================================================================

# data processing for sleep analysis 
df_sa_original = pd.read_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv')

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
df_sa_original.to_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv', index=False)  # save to new csv file
df_sa_filtered = df_sa_original[df_sa_original['onehot_encoded_value'] != 2]

# Now we can work with the date time 

# Read the new processed csv file --- change the names for df here to represent sleep analysis -> df_sa
df_sa = pd.read_csv('F:\FALL 2024\Senior-Project-Modeling-Noah\data\SleepAnalysis_label_data.csv')
# convert to datetime 
df_sa['start'] = pd.to_datetime(df_sa['start'])
df_sa_filtered = df_sa_filtered[df_sa_filtered['start'] >= '2023-06-01']

#the datetime values will be used 
df_sa_filtered.set_index('start', inplace=True)

if df_sa_filtered.index.duplicated().any():
    df_sa_filtered = df_sa_filtered[~df_sa_filtered.index.duplicated(keep='first')]

start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
#normalize them to a constant frequency 
common_time = pd.date_range(start=start_time, 
                            end=end_time, 
                            freq='min')

#align values with the times 
sleep_analysis_interpolated = df_sa_filtered['onehot_encoded_value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_sleep_analysis_df = pd.DataFrame({
    'onehot_encoded_value': sleep_analysis_interpolated
})


processed_sleep_analysis = aligned_sleep_analysis_df.to_numpy().flatten()

print(processed_sleep_analysis)

print("Length of common_time:", len(common_time))
print("Length of processed_sleep_analysis:", len(processed_sleep_analysis))


# =================================================================================================================
# =================================================================================================================
# ================================================    PLOTTING GRAPH     ==========================================
# =================================================================================================================
# =================================================================================================================

'''
# Align sleep data with Kalman filter steps
aligned_sleep_time = sleep_time_sans_nan[:612655]
aligned_sleep_analysis = sleep_analysis_sans_nan[:612655]

# Time steps from Kalman filter
time_steps = range(307242)

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot sleep time on the primary y-axis
ax1.plot(time_steps, aligned_sleep_time, color='blue', label='Sleep Time')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Sleep Duration (min)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot sleep types on the secondary y-axis
ax2 = ax1.twinx()
ax2.plot(time_steps, aligned_sleep_analysis, color='orange', label='Sleep Type')
ax2.set_ylabel('Sleep Type (Encoded)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Adding title and legend
plt.title('Sleep Patterns Aligned with Kalman Filter Time Steps')
fig.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(common_time, processed_sleep_analysis, label='Sleep States (Smoothed)')
plt.xlabel('Time')
plt.ylabel('Sleep Type (Encoded)')
plt.title('Original Sleep Analysis Data Before Interpolation')
plt.legend()
plt.show()
'''