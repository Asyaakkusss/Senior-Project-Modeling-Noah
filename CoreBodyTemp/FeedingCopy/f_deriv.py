import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import os
import sys
col_to_extract = "value"
import pandas as pd 
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
# sys.path.append("./SleepCycle")
sys.path.append("../../SleepCycle/")
from data_processing import process_categorical_data, process_numerical_data
from scipy.optimize import curve_fit


#preprocess the data 
# phys_rate_csv_string = "data/PhysicalEffort.csv"
# basal_rate_csv_string = "data/BasalEnergyBurned.csv"
phys_rate_csv_string = "../../data/PhysicalEffort.csv"
basal_rate_csv_string = "../../data/BasalEnergyBurned.csv"
col_interest = 'start'
processed_phys_rate = process_numerical_data(phys_rate_csv_string, col_interest)
processed_basal_rate = process_numerical_data(basal_rate_csv_string, col_interest)

#the three arrays have null values, so we crop the nulls out to leave as much valid data as possible  
processed_phys_rate = processed_phys_rate[612:]
processed_basal_rate = processed_basal_rate[612:]

#thr three arrays are now different lengths, so we find the minimum length and cut off the maximum index of an array at that minimum length 
min_length = min(len(processed_phys_rate), len(processed_basal_rate))

#create the index values for the p matrix by finding covariance between the three arrays 
processed_basal_rate = processed_basal_rate[:min_length]
processed_phys_rate = processed_phys_rate[:min_length]

time = np.arange(0, len(processed_basal_rate))

#Subtract 100 to find crossings around y=12.5
phys_rate_shifted = processed_phys_rate - 2.5

# Find hundred-crossings (where the data crosses 100, or the middle of one oscillation)
hundred_crossings = np.where(np.diff(np.sign(phys_rate_shifted)))[0]

# Number of oscillations is approximately half the zero-crossings
num_oscillations = len(hundred_crossings) // 2

print("Estimated number of oscillations", num_oscillations)
# Parameters for physical rate. we are forcing the fit bc what the fuck. 
N_h = len(time) 
f_h = 270
t_h = time  
#this is the estimated equation for physical effort rate 
y = 5.5 * np.cos(2 * np.pi * f_h * t_h / N_h) + 7


#parameters for basal metabolic rate
N_b= len(time)
f_b = 368
t_b= time 

#this is the estimate equation for basal metabolic rate 
y_1 = np.abs(24 * np.sin(2 * np.pi * f_b * t_b/N_b))

# Plot the original and reconstructed signals
plt.figure(figsize=(12, 6))
plt.plot(time, processed_phys_rate, label="Heart Rate (Original)")
#plt.plot(time, processed_basal_rate, label="BMR (Original)")
plt.plot(t_h, y, label="physical effort fit")
#plt.plot(t_b, y_1, label="BMR fit")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.title("Original vs. Reconstructed Signals for Sleep")
plt.savefig("correlation_sleep.png")
# plt.show()



#now...for finding the actual cells in the matrix.


#influence of HR w respect to BMR...aka the second row and fourth column. derivatives. 

pe = 5.5 * np.cos(2 * np.pi * f_h * t_h / N_h) + 7
bmr = np.abs(24 * np.sin(2 * np.pi * f_b * t_b/N_b))

#find gradient of each 
pe_gradient = np.gradient(pe, time)
bmr_gradient = np.gradient(bmr, time)

#find derivative of pe w respect to BMR
matrix_cell = np.mean(pe_gradient/bmr_gradient)

#find derivative of BMR w respect to pe
matrix_cell_1 = np.mean(bmr_gradient/pe_gradient)
print()
print("pe w respect to BMR", matrix_cell)
print("BMR w respect pe", matrix_cell_1)
#noah i love you pls figure out which cell is which. idk which one is hr w respect to bmr and which one is bm w respect to hr. as in cell (2, 3) vs (3, 2). 
#non matlab order this is matlab slander. 
# i have made the decision to make the 1-3 of the first row and columns zero becuase we simply are just girls and don't know (fairy and sparkle emojis)
#just slap dt in front of the constants I have produced. its not much but it is honest work 

plt.show()