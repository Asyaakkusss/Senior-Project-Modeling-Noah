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

# Start off with Asya's my_kalman_filter.py steps: load, convert from array to integer, then process the data

# Load in data from PureSleepTime.csv, HKCategoryTypeIdentifierSleepAnalysis.csv, BasalMetabolicRate.csv
with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_analysis = np.array(column_data) 


with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_time = np.array(column_data)

with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\BasalEnergyBurned.csv', 'r') as file:
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


# ============================================================================================================================
# ============================================================================================================================

# data processing for sleep analysis 
df_sa_original = pd.read_csv('data/HKCategoryTypeIdentifierSleepAnalysis.csv')

# First we are adding the column with the one hot encoded value to 'quantify' the sleep analysis data

# Category mapping for the values in the csv
category_mapping = {
    "HKCategoryValueSleepAnalysisInBed": 1,
    "HKCategoryValueSleepAnalysisAsleepREM": 2,
    "HKCategoryValueSleepAnalysisAsleepDeep": 3,
    "HKCategoryValueSleepAnalysisAsleepCore": 4,
    "HKCategoryValueSleepAnalysisAwake": 5,
    "HKCategoryValueSleepAnalysisAsleepUnspecified": 0,
}

# Map categories to numeric valuesby one hot encoding
df_sa_original['onehot_encoded_value'] = df_sa_original['value'].map(category_mapping)
df_sa_original.to_csv('data/HKCategoryTypeIdentifierSleepAnalysis_processed.csv', index=False)  # save to new csv file

# Now we can work with the date time 

# Read the new processed csv file --- change the names for df here to represent sleep analysis -> df_sa
df_sa = pd.read_csv('data/HKCategoryTypeIdentifierSleepAnalysis_processed.csv')
# convert to datetime 
df_sa['start'] = pd.to_datetime(df_sa['time'])

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

'''
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
sleep_analysis_interpolated = df_sa['value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_sleep_analysis_df = pd.DataFrame({
    'value': sleep_analysis_interpolated
})

processed_sleep_analysis = aligned_sleep_analysis_df.to_numpy().flatten()

# ====================================================================================================================

# data processing for basal metabolic rate 
df_be = pd.read_csv("data/BasalEnergyBurned.csv")

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



#print(processed_basal_rate)
#print(processed_sleep_analysis)
#print(processed_sleep_time)



final_P = np.array([
    [7, 0,            0,            0         ],
    [0, 98.18160966, -34.72900601, -1.22780453],
    [0, -34.72900601, 470.32652132, 3.1424907],
    [0, -1.22780453,  3.1424907,    3.33721444],
])

# Initial state X
X = np.array([
    [97],
    [np.mean(processed_basal_rate[650:])],
    [np.mean(processed_heart_rate[650:])],
    [np.mean(processed_respiratory[650:])],
]) 

# Process model matrix F
dt = 1  # 1 second time step
F = np.array([
    [1, 0, dt, 0.5*dt**2],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

# Measurement noise covariance matrix R
R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Two options for Q (process noise covariance)
Q_filterpy = Q_discrete_white_noise(dim=4, dt=1., var=7)
Q_manual = np.array([
    [0.7, 0, 0, 0],
    [0, 9.818160966, 0, 0],
    [0, 0, 47.032652132, 0],
    [0, 0, 0, 0.333721444],
])

# Measurement matrix H (maps state to measurement)
H = np.array([
    [0, 1, 0, 0],  # basal rate mapping
    [0, 0, 1, 0],  # heart rate mapping
    [0, 0, 0, 1]   # respiratory rate mapping
])

# Initialize Kalman filter
def initialize_kalman_filter(X, P, R, Q, F, H):
    kf = KalmanFilter(dim_x=4, dim_z=3)
    kf.x = X
    kf.P = P
    kf.R = R
    kf.Q = Q
    kf.F = F
    kf.H = H
    return kf

kf = initialize_kalman_filter(X, final_P, R, Q_manual, F, H)

# Load interpolated data
interpolated = []
dates = []

with open("interpolated_sleep_analysis.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        interpolated.append(float(row[1]))
        dates.append(pd.to_datetime(row[0]))

S = iter(interpolated)

# Use Kalman filter on data
xs = []
xxs = []
z = next(S, None)
while z is not None:
    kf.predict()
    kf.update([z])

    xs.append(kf.x[0])
    xxs.append(kf.x[1])

    z = next(S, None)

# Plot results
sns.lineplot(x=range(len(xs)), y=xs)
'''