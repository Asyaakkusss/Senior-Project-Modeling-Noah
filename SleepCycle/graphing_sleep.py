# Imports
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
import data_processing as dp


basal_rate_sans_nan_1D, sleep_analysis_sans_nan_1D, sleep_time_sans_nan_1D = dp.convert_1D()

#sys.path.append("F:\FALL 2024\Senior-Project-Modeling-Noah\CoreBodyTemp")
#from my_kalman_filter import n_steps
'''
# Start off with Asya's my_kalman_filter.py steps: load, convert from array to integer, then process the data

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


# =================================================================================================================
# =================================================================================================================
# ================================================    PLOTTING GRAPH     ==========================================
# =================================================================================================================
# =================================================================================================================


# Align sleep data with Kalman filter steps
#used to be 612655
amount_of_time = 1440

aligned_sleep_time = sleep_time_sans_nan_1D[:amount_of_time]
aligned_sleep_analysis = sleep_analysis_sans_nan_1D[:amount_of_time]

# Time steps from Kalman filter
time_steps = range(307242)

#plt.plot(basal_rate_sans_nan_1D[:amount_of_time], label='Basal Rate', alpha=0.7, linestyle='-')
plt.plot(aligned_sleep_analysis, label='Sleep Analysis', alpha=0.7, linestyle='--')
#plt.plot(aligned_sleep_time, label='Sleep Time', alpha=0.7, linestyle=':')
plt.title('Plot of Sleep Analysis(First 1440 Minutes)')
plt.ylabel('Values')
plt.legend()
plt.show()
'''
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
plt.plot(time_steps, sleep_analysis_sans_nan_1D, label='Sleep States (Smoothed)')
plt.xlabel('Time')
plt.ylabel('Sleep Type (Encoded)')
plt.title('Original Sleep Analysis Data Before Interpolation')
plt.legend()
plt.show()
'''
