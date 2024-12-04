import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import os
import sys
col_to_extract = "value"
import pandas as pd 
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
sys.path.append("/Users/monugoel/Desktop/CSDS_395/SleepCycle")
from data_processing import process_categorical_data, process_numerical_data, calc_R
from datetime import datetime


#extract heart rate data 
home_dir = "/Users/monugoel/Desktop/CSDS_395/"
#Asya change this to: /home/asyaakkus/Senior-Project-Modeling-Noah/

'''
we find the P matrix by taking the variance of each of the arrays at consistent time stamps. for now, we will just focus on 
respiratory rate and heart rate, since I am unsure whether or not VO is salvageable data point. There is not enough data 
and it is too spread out. I am going to use basal energy burned instead. 
'''

'''
We need to truncate the BE and HR arrays to be the same size as RR. With HR, it should be straightforward because both the 
measurements from the watch are taken in count/min. 

The timestamps don't align and this is a big problem. We have to probably use the interpolate method in pandas to make this 
work at all. 

RR first timestamp: [[Timestamp('2023-07-07 01:08:27-0400', tz='UTC-04:00') 17.0]
RR last timestamp:  [Timestamp('2024-09-05 08:27:27-0400', tz='UTC-04:00') 11.0]]


'''
#preprocess the data 
resp_rate_csv_string = os.path.join(home_dir, "data/RespiratoryRate.csv")
heart_rate_csv_string = os.path.join(home_dir, "data/HeartRate.csv")
basal_rate_csv_string = os.path.join(home_dir, "data/BasalEnergyBurned.csv")
col_interest = 'start'
processed_respiratory, time_series_resp = process_numerical_data(resp_rate_csv_string, col_interest)
processed_heart_rate, time_series_hr = process_numerical_data(heart_rate_csv_string, col_interest)
processed_basal_rate, time_series_bsl = process_numerical_data(basal_rate_csv_string, col_interest)

#the three arrays have null values, so we crop the nulls out to leave as much valid data as possible  
processed_respiratory = processed_respiratory[612:]
processed_heart_rate = processed_heart_rate[612:]
processed_basal_rate = processed_basal_rate[612:]
time_series_resp = time_series_resp[612:]

#thr three arrays are now different lengths, so we find the minimum length and cut off the maximum index of an array at that minimum length 
min_length = min(len(processed_respiratory), len(processed_heart_rate), len(processed_basal_rate))

#create the index values for the p matrix by finding covariance between the three arrays 
processed_basal_rate = processed_basal_rate[:min_length]
processed_heart_rate = processed_heart_rate[:min_length]
processed_respiratory = processed_respiratory[:min_length]
time_series_resp = time_series_resp[:min_length]

time_series_resp = [t.replace(tzinfo=None) for t in time_series_resp]


time_vals = range(0, len(processed_basal_rate))

# Number of time steps based on your data length
n_steps = len(processed_basal_rate)



# Create zs matrix: (n_steps, 3) where each row corresponds to measurements at one time step
zs = np.column_stack((processed_basal_rate, processed_heart_rate, processed_respiratory))


# Define initial matrices (already provided by you)

# Initial P matrix (state covariance matrix)
'''We set the covariances for the initial P to be 0 because the matrix has not converged yet. this is 
just an initialization and so the filter will update the covariances as the model propagates over time. 
The values were chosen based on the maximum reasonable numerical value for each as determined by a 
literature search and observation of processed data arrays. 
'''
P = np.array([
        [102, 0, 0, 0], #CBT
        [0, 22, 0, 0], #BMR
        [0, 0, 160, 0], #HR
        [0, 0, 0, 16], #RR
    ])

# Initial state X (based on means of X)
X = np.array([
    [99],
    [np.mean(processed_basal_rate[650:])],
    [np.mean(processed_heart_rate[650:])],
    [np.mean(processed_respiratory[650:])],
])

# Process model matrix F, equivalent to A in math equations
dt = 1  # 1 second time step


F = np.array([
    [1, 0, 0, 0],
    [0, 1, 3.31*np.cos(dt), 0.0001437*np.cos(dt)**2],
    [0, 0.276*np.cos(dt), 1, -2.713e-6*np.cos(dt)],
    [0, 1.835e-8, -6.9652e-10, 1],
])

def make_F(theta):
    return np.array([[1, 0, 0, 0],
    [0, -np.sin(theta), 3.31*np.cos(theta), 0.0001437*np.cos(theta)**2],
    [0, 0.276*np.sin(theta), 1, -2.713e-6*np.cos(theta)],
    [0, 1.835e-8*np.cos(theta), -6.9652e-10*np.cos(theta), 1]])

# Measurement noise covariance matrix R. Basically what we did for P before. Little goof 
R = calc_R([processed_basal_rate, processed_heart_rate, processed_respiratory])
print(R)

# finding variance for R

'''
var_bmr = processed_basal_rate.var()
var_rr = processed_respiratory.var()
var_hr = processed_heart_rate.var()

R = np.diag([var_bmr, var_hr, var_rr])'''

# Two options for Q (process noise covariance)
Q_filterpy = Q_discrete_white_noise(dim=4, dt=1., var=7)
Q_manual = np.array([
    [0.5, 0, 0, 0],
    [0, 0.2, 0, 0],
    [0, 0, 0.3, 0],
    [0, 0, 0, 0.2],
])

# Measurement matrix H (maps state to measurement)
H = np.array([
    [0, 1, 0, 0],  # basal rate mapping
    [0, 0, 1, 0],  # heart rate mapping
    [0, 0, 0, 1]   # respiratory rate mapping
])

# Kalman filter initialization
def initialize_kalman_filter(X, P, R, Q, F, H):
    kf = KalmanFilter(dim_x=4, dim_z=3)
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
def run_kalman_filter_cbt(X, P, R, Q, F, H, zs, n_steps):
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
xs, Ps, residuals = run_kalman_filter_cbt(X, P, R, Q_filterpy, F, H, zs, n_steps)

# xs now contains state estimates, including core body temperature estimates over time
print(type(xs))
print(np.shape(xs))


xs_reshaped = xs.reshape(612655, 4)
#Removed this line because it creates an inordinately large file
#np.savetxt(os.path.join(home_dir, "predictions_cbt.csv"), xs_reshaped, delimiter=",")

xs_cbt = xs_reshaped[:15000, 0]
time_series_resp = time_series_resp[:15000]

ys_cbt = np.arange(len(xs_cbt))

np.savetxt(os.path.join(home_dir,"Ensemble/cbtarray.csv"),xs_cbt)

# Calculate Standard Deviation of Residuals and MSE
residual_std = np.std(residuals, axis=0)
mse = np.mean(residuals**2, axis=0)


start_time = datetime(2023, 7, 8, 0, 0)  # Example start time: 05:00
end_time = datetime(2023, 7, 9, 0, 0)
indices = [i for i, t in enumerate(time_series_resp) if start_time <= t <= end_time]
time_series_df = pd.Series(time_series_resp)
time_series = time_series_df.iloc[indices]
time_series = time_series.tolist()

xs_cbt = xs_cbt[indices]
#time_series = pd.date_range(start="00:01", end="23:59", periods=len(xs_cbt))

#plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=120))  # Set ticks every 30 minutes
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

locator = mdates.HourLocator(interval=2)  # ticks every 2 hours
plt.gca().xaxis.set_major_locator(locator)
formatter = mdates.DateFormatter('%H:%M')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().set_xlim(start_time, end_time)

residual_std_cbt = residual_std[:, 0]  # Extract standard deviation for CBT
mse_cbt = mse[:, 0]  # Extract MSE for CBT

print("Standard Deviation of Residuals:", residual_std_cbt)
print("Mean Squared Error (MSE):", mse_cbt)


plt.plot(time_series, xs_cbt, color = "#003071", label='Predicted CBT')
plt.title("Subject\'s Predicted Core Body Temperature Over 24 Hours")
plt.xlabel('Time (Hours)')
plt.ylabel('CBT Estimate (°F)')
#plt.legend()
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
plt.savefig('my_plot_kal_with_residuals.png')
plt.show()
