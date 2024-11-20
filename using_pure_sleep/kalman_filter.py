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
with open(r'/home/asyaakkus/Senior-Project-Modeling-Noah/data/HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
# with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\HKCategoryTypeIdentifierSleepAnalysis.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_analysis = np.array(column_data) 

with open(r'/home/asyaakkus/Senior-Project-Modeling-Noah/data/PureSleepTime.csv', 'r') as file:
#with open(r'F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv', 'r') as file:
    reader = csv.DictReader(file)
    column_data = [row[col_to_extract] for row in reader]

sleep_time = np.array(column_data)

with open(r'/home/asyaakkus/Senior-Project-Modeling-Noah/data/BasalEnergyBurned.csv', 'r') as file:
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
df_st = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/data/PureSleepTime.csv")
#df_st = pd.read_csv("F:\FALL 2024\Senior-Project-Modeling-Noah\data\PureSleepTime.csv")

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
sleep_analysis_interpolated = df_sa['onehot_encoded_value'].reindex(common_time).interpolate()

#create a dataframe with start and value columns 
aligned_sleep_analysis_df = pd.DataFrame({
    'onehot_encoded_value': sleep_analysis_interpolated
})

processed_sleep_analysis = aligned_sleep_analysis_df.to_numpy().flatten()
all_nan_sleep_analysis = np.isnan(processed_sleep_analysis)
sleep_analysis_sans_nan = processed_sleep_analysis[~all_nan_sleep_analysis]
# print(processed_sleep_analysis)

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
all_nan_basal_rate = np.isnan(processed_basal_rate)
basal_rate_sans_nan = processed_basal_rate[~all_nan_basal_rate]

# =============================================================================================================

# Currently, becuase the different data sets have different number of nans, the data sizes are different, 
# So we use the minimum length of all the data and use that size for all the arrays of data

# Ensuring consistent length of the arrays
min_length = min(len(basal_rate_sans_nan), len(sleep_analysis_sans_nan), len(sleep_time_sans_nan))
basal_rate_sans_nan = basal_rate_sans_nan[:min_length]
sleep_analysis_sans_nan = sleep_analysis_sans_nan[:min_length]
sleep_time_sans_nan = sleep_time_sans_nan[:min_length]

# Convert to 1D arrays
basal_rate_sans_nan = np.array(basal_rate_sans_nan).flatten()
sleep_analysis_sans_nan = np.array(sleep_analysis_sans_nan).flatten()
sleep_time_sans_nan = np.array(sleep_time_sans_nan).flatten()

# It looks beautiful after this :)


# =============================== DEBUGGING PURPOSES ================================
#print(processed_basal_rate)
#print(processed_sleep_analysis)
#print(processed_sleep_time)
#print(basal_rate_sans_nan, len(basal_rate_sans_nan))
#print(sleep_time_sans_nan, len(sleep_time_sans_nan))
#print(sleep_analysis_sans_nan, len(sleep_analysis_sans_nan))


# =============================================================================================================
# =============================================================================================================
# =============================================================================================================
# ============================================ We start working on the matrices ===============================

# Creation of P matrix
unified_array = np.array([basal_rate_sans_nan, sleep_analysis_sans_nan, sleep_time_sans_nan])
P_threebythree = np.cov(unified_array)


# Number of time steps based on the length of data being used
n_steps = len(basal_rate_sans_nan)

# Create zs matrix: (n_steps, 3) where each row corresponds to measurements at one time step
zs = np.column_stack((basal_rate_sans_nan, sleep_analysis_sans_nan, sleep_time_sans_nan))


# Define initial matrices (already provided by you)

# Initial P matrix (state covariance matrix)

# =================================== this is where my understanding ends =========================================


#please note that the 0.16 for the total rhythm variance is currently just a placeholder. we need to figure out the average
#variance of the sleeper from his data. For example, a sample when he falls asleep at 10:00 would be 22, at 6:00 AM would be 6, 
#basically the variance based on military time. 
#this P is taken from the P_threebythree array 
final_P = np.array([
    [0.16, 0,            0,            0         ],
    [1.10624371e+02, 1.61602628e-02, 4.40134994e-02], 
 [1.61602628e-02, 1.10291078e-02, 1.27332509e-02], 
 [4.40134994e-02, 1.27332509e-02, 2.86459969e-02], 
])

# Initial state X. based on the average of the arrays sans all the nans 
#the hidden variable is initialized as 24 because this is the average length of a sleep-wake cycle 
X = np.array([
    [24],
    [np.mean(basal_rate_sans_nan)],
    [np.mean(sleep_analysis_sans_nan)],
    [np.mean(sleep_time_sans_nan)],
]) 

# Process model matrix F
dt = 1  # 1 second time step as created by processing the data 
F = np.array([
    [1, 0, dt, 0.5*dt**2],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

# From ashley's branch --- implementation of rotation matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta),  np.cos(theta), 0, 0],
                     [0,0,1,0],
                     [0,0,0,1]])

def rotation_matrix_4d_xy_xw(theta_xy, theta_xw):
    """Returns a combined 4x4 rotation matrix for rotation in the xy-plane and xw-plane."""
    R_xy = np.array([[np.cos(theta_xy), -np.sin(theta_xy), 0, 0],
                     [np.sin(theta_xy), np.cos(theta_xy),  0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    R_xw = np.array([[np.cos(theta_xw), 0, 0, -np.sin(theta_xw)],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [np.sin(theta_xw), 0, 0, np.cos(theta_xw)]])
    
    return R_xy @ R_xw  # Combine the two rotations

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
    [0, 0, 1, 0],  # should be sleep analysis mapping
    [0, 0, 0, 1]   # should be sleep time mapping
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

omega_xy = np.pi/6
omega_xw = np.pi/8
omega = np.pi/6
# Kalman filter loop: Predict and update steps | here I am also finding the residuals inside the Kalman filter loop
def run_kalman_filter(X, P, R, Q, F, H, zs, n_steps):
    kf = initialize_kalman_filter(X, P, R, Q, F, H)
    F_rotation = rotation_matrix(omega)
    kf_rot = initialize_kalman_filter(X, P, R, Q, F_rotation, H)
    # Arrays to store state estimates and covariances
    xs, cov = [], []
    xs_rot = []
    residuals = []
    
    for i in range(n_steps):
        theta_xy = omega_xy * i * dt
        theta_xw = omega_xw * i * dt
        #kf_rot.F = rotation_matrix_4d_xy_xw(theta_xy, theta_xw)

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
xs, Ps, residuals = run_kalman_filter(X, final_P, R, Q_filterpy, F, H, zs, n_steps)

# xs now contains state estimates, (including core body temperature estimates over time) --- what is this supposed to show now then? 
print(type(xs))
print(np.shape(xs))


# Extract the estimated states
time_steps = range(len(xs))
estimated_basal_rate = xs[:, 0]  # State 1 (basal metabolic rate)
estimated_sleep_analysis = xs[:, 1]  # State 2 (sleep analysis)
estimated_sleep_time = xs[:, 2]  # State 3 (sleep time)

# Plot state estimates
plt.figure(figsize=(10, 6))
plt.plot(time_steps, estimated_basal_rate, label="Estimated Basal Rate")
plt.plot(time_steps, estimated_sleep_analysis, label="Estimated Sleep Analysis")
plt.plot(time_steps, estimated_sleep_time, label="Estimated Sleep Time")
plt.xlabel("Time Steps")
plt.ylabel("State Estimates")
plt.title("Kalman Filter State Estimates Over Time")
plt.legend()
plt.grid()
plt.savefig('sleepfactors.png')
plt.show()

# Compare original and estimated values
plt.figure(figsize=(10, 6))
plt.plot(time_steps, zs[:, 0], label="Original Basal Rate", linestyle="dashed")
plt.plot(time_steps, estimated_basal_rate, label="Estimated Basal Rate")
plt.xlabel("Time Steps")
plt.ylabel("Values")
plt.title("Original vs Estimated Basal Rate")
plt.legend()
plt.grid()
plt.show()





'''

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
