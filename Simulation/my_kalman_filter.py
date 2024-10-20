'''
Im going to need a state vector like so: [CBT HR ST]

we need parameters describing how skin temp and heart rate impact core body temp 

Process model: 
A = 
[1 alpha beta]
[0 1     0]
[0 0     1]

x_t+1 = A*xt + w_t

Measurement model: H matrix with heart rate and skin temperature from state vector 

H = [0 1 0]
    [0 0 1]

Q covariance (for process model): process noise 
[Q_CBT 0 0]
[0 Q_HR 0]
[0 0 Q_ST]

R covariance (for measurement model): variances for heart rate and skin temp 
R = [R_HR 0]
    [0 R_ST]


'''
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import os
col_to_extract = "value"
import pandas as pd 
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

#extract respiratory rate data

#extract heart rate data 
home_dir = "/Users/monugoel/Desktop/CSDS_395"
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/HeartRate.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    column_data = [row[col_to_extract] for row in reader]

heart_rate = np.array(column_data)


#extract respiratory rate data 
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/RespiratoryRate.csv', 'r') as file: 
    reader = csv.DictReader(file)

    column_data = [row[col_to_extract] for row in reader]

respiratory_rate = np.array(column_data)


#extract basal energy burned data 
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/BasalEnergyBurned.csv', 'r') as file: 
    reader = csv.DictReader(file)

    column_data = [row[col_to_extract] for row in reader]

basal_energy_burned = np.array(column_data)



#convert the arrays to integers 
def convert_to_integer(array): 
    return np.array(array).astype(float)

RR = convert_to_integer(respiratory_rate)
BE = convert_to_integer(basal_energy_burned)
HR = convert_to_integer(heart_rate)


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

#data processing for respiratory rate 
df = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/RespiratoryRate.csv")

#convert to datetime 
df['start'] = pd.to_datetime(df['start'])

#the datetime values will be used 
df.set_index('start', inplace=True)

#normalize them to a constant frequency 
common_time = pd.date_range(start=df.index.min(), end=df.index.max(), freq='min')

#align values with the times 
respir_interpolated = df['value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_rr_df = pd.DataFrame({
    'value': respir_interpolated
})

processed_respiratory = aligned_rr_df.to_numpy().flatten()


#data processing for heart rate 
df = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/HeartRate.csv")

#convert to datetime 
df['start'] = pd.to_datetime(df['start'])

#the datetime values will be used 
df.set_index('start', inplace=True)

if df.index.duplicated().any():
    df = df[~df.index.duplicated(keep='first')]

start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
#normalize them to a constant frequency 
common_time = pd.date_range(start=start_time, 
                            end=end_time, 
                            freq='min')

#align values with the times 
heartrate_interpolated = df['value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_hr_df = pd.DataFrame({
    'value': heartrate_interpolated
})

processed_heart_rate = aligned_hr_df.to_numpy().flatten()

#data processing for basal metabolic rate 
df = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/BasalEnergyBurned.csv")

#convert to datetime 
df['start'] = pd.to_datetime(df['start'])

#the datetime values will be used 
df.set_index('start', inplace=True)

if df.index.duplicated().any():
    df = df[~df.index.duplicated(keep='first')]

start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
#normalize them to a constant frequency 
common_time = pd.date_range(start=start_time, 
                            end=end_time, 
                            freq='min')

#align values with the times 
basal_interpolated = df['value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_basal_df = pd.DataFrame({
    'value': basal_interpolated
})

processed_basal_rate = aligned_basal_df.to_numpy().flatten()

#creation of P matrix values 
unified_array = np.array([processed_basal_rate[650:], processed_heart_rate[650:], processed_respiratory[650:]])
P_threebythree = np.cov(unified_array)

# Preprocess your data into arrays
basal_rate_data = np.array(processed_basal_rate[650:])
heart_rate_data = np.array(processed_heart_rate[650:])
respiratory_data = np.array(processed_respiratory[650:])

# Number of time steps based on your data length
n_steps = len(basal_rate_data)

# Create zs matrix: (n_steps, 3) where each row corresponds to measurements at one time step
zs = np.column_stack((basal_rate_data, heart_rate_data, respiratory_data))

# Define initial matrices (already provided by you)

# Initial P matrix (state covariance matrix)
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

# Process model matrix F, equivalent to A in math equations
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

# Kalman filter loop: Predict and update steps
def run_kalman_filter(X, P, R, Q, F, H, zs, n_steps):
    kf = initialize_kalman_filter(X, P, R, Q, F, H)
    
    # Arrays to store state estimates and covariances
    xs, cov = [], []
    
    for i in range(n_steps):
        kf.predict()  # Predict the next state
        z = zs[i]     # Get the measurements for this time step
        kf.update(z)  # Update with the measurement
        
        xs.append(kf.x)  # Store the state estimate
        cov.append(kf.P) # Store the covariance matrix
    
    # Convert results to numpy arrays for easy handling
    xs = np.array(xs)
    cov = np.array(cov)
    
    return xs, cov

# Run the Kalman filter with your data
xs, Ps = run_kalman_filter(X, final_P, R, Q_filterpy, F, H, zs, n_steps)

# xs now contains state estimates, including core body temperature estimates over time
print(xs)

print(np.shape(xs))

xs_reshaped = xs.reshape(613230, 4)

xs_cbt = xs_reshaped[:1440, 0]
ys_cbt = np.arange(len(xs_cbt))
print(len(xs_cbt))

plt.plot(xs_cbt, ys_cbt)
plt.savefig('my_plot_kal.png')