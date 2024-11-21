
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
# Ensuring consistent length of the arrays
min_length = min(len(basal_rate_sans_nan), len(sleep_analysis_sans_nan), len(sleep_time_sans_nan))
basal_rate_sans_nan = basal_rate_sans_nan[:min_length]
sleep_analysis_sans_nan = sleep_analysis_sans_nan[:min_length]
sleep_time_sans_nan = sleep_time_sans_nan[:min_length]

# Convert to 1D arrays
basal_rate_sans_nan = np.array(basal_rate_sans_nan).flatten()
sleep_analysis_sans_nan = np.array(sleep_analysis_sans_nan).flatten()
sleep_time_sans_nan = np.array(sleep_time_sans_nan).flatten()
