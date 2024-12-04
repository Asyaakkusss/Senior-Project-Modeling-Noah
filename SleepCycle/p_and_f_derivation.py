
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import LabelEncoder
from data_processing import process_categorical_data, process_numerical_data

#________________________________________________________data analysis cleaned up______________________________________________________________#
#strings for data analysis 
# sleep_analysis_categorical_fp = '/home/asyaakkus/Senior-Project-Modeling-Noah/data/HKCategoryTypeIdentifierSleepAnalysis.csv'
# sleep_analysis_fp = 'data/HKCategoryTypeIdentifierSleepAnalysis_processed.csv'
# pure_sleep_time_fp = '/home/asyaakkus/Senior-Project-Modeling-Noah/data/PureSleepTime.csv'
# basal_energy_fp = '/home/asyaakkus/Senior-Project-Modeling-Noah/data/BasalEnergyBurned.csv'

sleep_analysis_categorical_fp = '../data/HKCategoryTypeIdentifierSleepAnalysis.csv'
sleep_analysis_fp = '../data/HKCategoryTypeIdentifierSleepAnalysis_processed.csv'
pure_sleep_time_fp = '../data/PureSleepTime.csv'
basal_energy_fp = '../data/BasalEnergyBurned.csv'

#array of category mapping 
# Category mapping for the values in the csv
category_mapping = {
    "HKCategoryValueSleepAnalysisInBed": 1,
    "HKCategoryValueSleepAnalysisAsleepREM": 2,
    "HKCategoryValueSleepAnalysisAsleepDeep": 3,
    "HKCategoryValueSleepAnalysisAsleepCore": 4,
    "HKCategoryValueSleepAnalysisAwake": 5,
    "HKCategoryValueSleepAnalysisAsleepUnspecified": 0,
    }

sleep_time = process_numerical_data(pure_sleep_time_fp, "time")


# Load the CSV file into a DataFrame
df = pd.read_csv(pure_sleep_time_fp)

# Extract a single column as a NumPy array (e.g., "ColumnName")
column_data = df['time'].to_numpy()




plt.figure(figsize=(10, 6))
plt.plot(range(len(column_data)), column_data, label="Raw Sleep Time")
plt.xlabel("Time Steps")
plt.ylabel("Pure Sleep Time per Second")
plt.title("Processed Sleep Time Data")
plt.legend()
plt.grid()
plt.savefig('puresleeptimevisual.png')
plt.show()
# Currently, becuase the different data sets have different number of nans, the data sizes are different, 
# So we use the minimum length of all the data and use that size for all the arrays of data

# Ensuring consistent length of the arrays

'''
min_length = min(len(basal_rate_sans_nan), len(sleep_analysis_sans_nan), len(sleep_time_sans_nan))
basal_rate_sans_nan = basal_rate_sans_nan[:min_length]
sleep_analysis_sans_nan = sleep_analysis_sans_nan[:min_length]
sleep_time_sans_nan = sleep_time_sans_nan[:min_length]

# Convert to 1D arrays
basal_rate_sans_nan = np.array(basal_rate_sans_nan).flatten()
sleep_analysis_sans_nan = np.array(sleep_analysis_sans_nan).flatten()
sleep_time_sans_nan = np.array(sleep_time_sans_nan).flatten()
'''