# imports
#home_dir = '/Users/monugoel/Desktop/CSDS_395/'
#asya change this to "/home/asyaakkus/Senior-Project-Modeling-Noah/"
home_dir = "../../"
import numpy as np 
import os
import csv 

import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates

from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import sys 
sys.path.append(os.path.join(home_dir, "SleepCycle"))
from data_processing import process_categorical_data, process_numerical_data, calc_R, calc_X
from datetime import datetime

#preprocess the data 
physical_csv_string = os.path.join(home_dir, "data/PhysicalEffort.csv")
basal_rate_csv_string = os.path.join(home_dir, "data/BasalEnergyBurned.csv")
col_interest = 'start'
processed_phys_rate, time_phys_rate  = process_numerical_data(physical_csv_string, col_interest)
processed_basal_rate, time_basal_rate  = process_numerical_data(basal_rate_csv_string, col_interest)

#the three arrays have null values, so we crop the nulls out to leave as much valid data as possible  
processed_phys_rate = processed_phys_rate[612:]
processed_basal_rate = processed_basal_rate[612:]
time_phys_rate = time_phys_rate[612:]
print (processed_phys_rate)

#thr three arrays are now different lengths, so we find the minimum length and cut off the maximum index of an array at that minimum length 
min_length = min(len(processed_phys_rate), len(processed_basal_rate))

#create the index values for the p matrix by finding covariance between the three arrays 
processed_basal_rate = processed_basal_rate[:min_length]
processed_phys_rate = processed_phys_rate[:min_length]
time_phys_rate = time_phys_rate[:min_length]
time_phys_rate = [t.replace(tzinfo=None) for t in time_phys_rate]

time_vals = range(0, len(processed_basal_rate))

# Number of time steps based on your data length
n_steps = len(processed_basal_rate)

# Create zs matrix: (n_steps, 3) where each row corresponds to measurements at one time step
zs = np.column_stack((processed_basal_rate, processed_phys_rate))

P = np.array([
        [1, 0, 0], #hunger cycle 
        [0, 22, 0], #BMR        [0, 22, 0]
        [0, 0, 12.5], #PE       [0, 0, 12.5]
    ])
# Initial state X (based on means of X)

X = np.array([
    [10],
    [np.mean(processed_basal_rate[:])],
    [np.mean(processed_phys_rate[:])],
])



# Process model matrix F, equivalent to A in math equations
dt = 1  # 1 second time step

# const_1 = 0.167489 default values
# const_2 = -1.52003
const_1 = 0.167489
const_2 = -1.52003
F = np.array([
    [1, 0, 0],
    [0, 1, const_1*np.cos(dt)],
    [0, const_2*np.cos(dt), 1],
])

def make_F(theta):
    return np.array([[1, 0, 0],
                     [0, 1, 0.167489*np.cos(theta)],
                     [np.cos(theta**2), -1.52003*np.sin(theta), 1],
    ])

# Measurement noise covariance matrix R. Basically what we did for P before. Little goof 
R = calc_R([processed_basal_rate, processed_phys_rate])
print(R)

# Two options for Q (process noise covariance)
Q_filterpy = Q_discrete_white_noise(dim=3, dt=1., var=7)
print(Q_filterpy)
Q_manual = np.array([
    [50, 0, 0],
    [0, 20, 0],
    [0, 0, 30],
])

# Measurement matrix H (maps state to measurement)
H = np.array([
    [0, 0.5, 0.5],  # basal rate mapping 
    [0, 0.75, 1],  # phys rate mapping 
])

# H = np.array([
#     [0, 1, 0],  # basal rate mapping 
#     [0, 0, 1],  # phys rate mapping 
# ])

# Kalman filter initialization
def initialize_kalman_filter(X, P, R, Q, F, H):
    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = X
    kf.P = P
    kf.R = R
    kf.Q = Q
    kf.F = F
    kf.H = H
    return kf

omega_xy = np.pi/6
omega_xw = np.pi/8
omega = np.pi/6
# Kalman filter loop: Predict and update steps | here I am also finding the residuals inside the Kalman filter loop
def run_kalman_filter_hunger(X, P, R, Q, F, H, zs, n_steps):
    kf = initialize_kalman_filter(X, P, R, Q, F, H)
    F_rotation = make_F(omega)
    kf_rot = initialize_kalman_filter(X, P, R, Q_filterpy, F_rotation, H)
    # Arrays to store state estimates and covariances
    xs, cov = [], []
    xs_rot = []
    residuals = []
    
    for i in range(n_steps):
        theta_xy = omega_xy * i * dt
        theta_xw = omega_xw * i * dt

        kf.predict()  # Predict the next state
        kf_rot.predict()
        z = zs[i]     # Get the measurements for this time step
        kf.update(z)  # Update with the measurement
        kf_rot.update(z)

        xs.append(kf.x)  # Store the state estimate
        xs_rot.append(kf_rot.x)
        cov.append(kf.P) # Store the covariance matrix

        # Calculate residuals (difference between measurement and prediction)
        predicted_measurement = H @ kf.x  # Predicted measurement
        residual = z - predicted_measurement
        residuals.append(residual)
    
    # Convert results to numpy arrays for easy handling
    xs = np.array(xs)
    cov = np.array(cov)
    xs_rot = np.array(xs_rot)
    residuals = np.array(residuals) 

    return xs_rot, cov, residuals

# Run the Kalman filter with your data
xs, Ps, residuals = run_kalman_filter_hunger(X, P, R, Q_filterpy, F, H, zs, n_steps)

# xs now contains state estimates, including hunger estimates over time
print(type(xs))
print(np.shape(xs))


xs_reshaped = xs.reshape(302171, 3)

xs_cbt = xs_reshaped[:15000, 0]
time_phys_rate = time_phys_rate[:15000]
ys_cbt = np.arange(len(xs_cbt))
np.savetxt(os.path.join(home_dir, "Ensemble/hungerarray.csv"),xs_cbt)

# Calculate Standard Deviation of Residuals and MSE
residual_std = np.std(residuals, axis=0)
mse = np.mean(residuals**2, axis=0)

residual_std_cbt = residual_std[:, 0]  # Extract standard deviation for CBT
mse_cbt = mse[:, 0]  # Extract MSE for CBT

print("Standard Deviation of Residuals:", residual_std_cbt)
print("Mean Squared Error (MSE):", mse_cbt)

print(f"first time series: {time_phys_rate[0]}")
print(f"last time series: {time_phys_rate[-1]}")


start_time = datetime(2024, 2, 10, 0, 0)  # Example start time: 05:00
end_time = datetime(2024, 2, 11, 0, 0)
indices = [i for i, t in enumerate(time_phys_rate) if start_time <= t <= end_time]
time_series_df = pd.Series(time_phys_rate)
time_series = time_series_df.iloc[indices]
time_series = time_series.tolist()


xs_cbt = xs_cbt[indices]


plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=120))  # Set ticks every 30 minutes
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)


plt.plot(time_series, xs_cbt, label='Predicted Hunger Levels')
plt.title('Predicted Hunger over Time')
plt.xlabel('Time Steps')
plt.ylabel('Hunger Estimate')
plt.legend()
'''
# Plot Residuals
plt.subplot(2, 1, 2)  # Second plot in the grid
plt.plot(ys_cbt, residuals[:len(ys_cbt), 0], label='Residuals (CBT)')
plt.title('Residuals of Core Body Temperature Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Residual (Measurement - Prediction)')
plt.legend()
'''

# Show the final plot with both graphs
plt.tight_layout()
plt.savefig('feeding_output.png')
plt.show()
