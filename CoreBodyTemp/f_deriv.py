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
sys.path.append("/home/asyaakkus/Senior-Project-Modeling-Noah/SleepCycle")
from data_processing import process_categorical_data, process_numerical_data
from scipy.optimize import curve_fit


#preprocess the data 
resp_rate_csv_string = "data/RespiratoryRate.csv"
heart_rate_csv_string = "data/HeartRate.csv"
basal_rate_csv_string = "data/BasalEnergyBurned.csv"
col_interest = 'start'
processed_respiratory = process_numerical_data(resp_rate_csv_string, col_interest)
processed_heart_rate = process_numerical_data(heart_rate_csv_string, col_interest)
processed_basal_rate = process_numerical_data(basal_rate_csv_string, col_interest)

#the three arrays have null values, so we crop the nulls out to leave as much valid data as possible  
processed_respiratory = processed_respiratory[612:]
processed_heart_rate = processed_heart_rate[612:]
processed_basal_rate = processed_basal_rate[612:]

#thr three arrays are now different lengths, so we find the minimum length and cut off the maximum index of an array at that minimum length 
min_length = min(len(processed_respiratory), len(processed_heart_rate), len(processed_basal_rate))

#create the index values for the p matrix by finding covariance between the three arrays 
processed_basal_rate = processed_basal_rate[:min_length]
processed_heart_rate = processed_heart_rate[:min_length]
processed_respiratory = processed_respiratory[:min_length]

time = np.arange(0, len(processed_basal_rate))
'''
fig, ax = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns

# Plot the histogram for the first dataframe
ax[0].plot(time, processed_basal_rate)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Basal Metabolic Rate')
ax[0].grid(True)

# Plot the histogram for the second dataframe
ax[1].plot(time, processed_heart_rate)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Heart Rate (BPM)')
ax[1].grid(True)

# Plot the histogram for the third dataframe 
ax[2].plot(time, processed_respiratory)
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Respiratory Rate (BPM)')
ax[2].grid(True)

plt.savefig("visualize_f_modeling.png")
'''
'''
each variable here influences one another. I have decided to start with that the bottom diagonal should be all 0's and the 
diagonal itself should be 1 because each variables influence itself in a constant sense. These values also don't change or fluctuate much, 
so the previous measurement for core body temp, HR, RR, and BMR will all be pretty similar in a negligible way. We could fit an equation though to 
the HR, RR, and BMR data we have and figure it out from there. 

Now, for the other two motherfuckers. There is an L shaped mf in the matrix on the lower right triangle that is as follows: 
1.influence of HR on BMR
2. influence of RR on BMR
3. influence of HR on RR
we will tackle these first and figure out connections 
'''

#Subtract 100 to find crossings around y=97.5
heart_rate_shifted = processed_heart_rate - 97.5

# Find hundred-crossings (where the data crosses 100, or the middle of one oscillation)
hundred_crossings = np.where(np.diff(np.sign(heart_rate_shifted)))[0]

# Number of oscillations is approximately half the zero-crossings
num_oscillations = len(hundred_crossings) // 2

print("Estimated number of oscillations", num_oscillations)
# Parameters for heart rate. we are forcing the fit bc what the fuck. 
N_h = len(time) 
f_h = 468
t_h = time  
#this is the estimated equation for heart rate 
y = 62.5 * np.cos(2 * np.pi * f_h * t_h / N_h) + 112.5

zero_crossings = np.where(heart_rate_shifted == 0)

#parameters for basal metabolic rate
N_b= len(time)
f_b = 368
t_b= time 

#this is the estimate equation for heart rate 
y_1 = np.abs(24 * np.sin(2 * np.pi * f_b * t_b/N_b))

#parameters 
# Plot the original and reconstructed signals
plt.figure(figsize=(12, 6))
#plt.plot(time, processed_heart_rate, label="Heart Rate (Original)")
#plt.plot(time, processed_basal_rate, label="BMR (Original)")
plt.plot(time, processed_respiratory, label="RR (Original)")
#plt.plot(t_h, y, label="heart rate fit")
#plt.plot(t_b, y_1, label="BMR fit")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.title("Original vs. Reconstructed Signals")
plt.savefig("correlation.png")


'''
now...for finding the actual cells in the matrix.
'''

#influence of HR w respect to BMR...aka the second row and fourth column. derivatives. 

hr = 62.5 * np.cos(2 * np.pi * f_h * t_h / N_h) + 112.5
bmr = np.abs(24 * np.sin(2 * np.pi * f_b * t_b/N_b))

#find gradient of each 

hr_gradient = np.gradient(hr, time)
bmr_gradient = np.gradient(bmr, time)

#find derivative of heart rate w respect to BMR
matrix_cell = np.mean(hr_gradient/bmr_gradient)

#find derivative of BMR w respect to HR
matrix_cell_1 = np.mean(bmr_gradient/hr_gradient)
print(matrix_cell)


matrix_cell_2
matrix_cell_3

matrix_cell_4
matrix_cell_5

matei

#noah i love you pls figure out which cell is which. idk which one is hr w respect to bmr and which one is bm w respect to hr. as in cell (2, 3) vs (3, 2). 
#non matlab order this is matlab slander. 
# i have made the decision to make the 1-3 of the first row and columns zero becuase we simply are just girls and don't know (fairy and sparkle emojis)
#just slap dt in front of the constants I have produced. its not much but it is honest work 