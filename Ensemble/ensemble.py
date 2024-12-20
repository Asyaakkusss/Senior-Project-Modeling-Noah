import sys 
import os
from datetime import datetime
#FOR ASYA TO EXECUTE
#sys.path.append("/home/asyaakkus/Senior-Project-Modeling-Noah/SleepCycle/")

#FOR NOAH TO EXECUTE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SleepCycle.data_processing import process_categorical_data, process_numerical_data, calc_R
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

#FOR NOAH TO EXECUTE:
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the CSV files
cbt_file_path = os.path.join(script_dir, "cbtarray.csv")
hun_file_path = os.path.join(script_dir, "hungerarray.csv")


# Load CSV into DataFrames
df_cbt = pd.read_csv(cbt_file_path)
df_hun = pd.read_csv(hun_file_path)
df_time = pd.read_csv(os.path.join(script_dir, "timearray.csv"), parse_dates=['Datetime'])
print(df_time['Datetime'].dtype)
#FOR ASYA TO EXECUTE
'''
# Load CSV into a DataFrame
df_cbt = pd.read_csv("cbtarray.csv")
df_hun = pd.read_csv("hungerarray.csv")
'''


# Convert to a NumPy array
arr_cbt = df_cbt.to_numpy().flatten()
arr_hun = df_hun.to_numpy().flatten()
arr_time = df_time.to_numpy().flatten()

print(type(arr_time[0]))

ys = np.arange(len(arr_cbt))

time_vals = range(0, len(arr_cbt))

# Number of time steps based on your data length
n_steps = len(arr_cbt)

# Create zs matrix: (n_steps, 3) where each row corresponds to measurements at one time step
zs = np.column_stack((arr_cbt, arr_hun))

P = np.array([
        [1, 0, 0], #ensemble
        [0, 97, 0], #cbt
        [0, 0, 10], #hunger
    ])

# Initial state X (based on means of X)
X = np.array([
    [0],
    [np.mean(arr_cbt)],
    [np.mean(arr_hun)],
])

# Process model matrix F, equivalent to A in math equations
dt = 1  # 1 second time step

F = np.array([
    [1, 0, 0],
    [0, 1, 0.167489*np.cos(dt)],
    [0, -1.52003*np.cos(dt), 1],
])

def make_F(theta):
    return np.array([[1, 0, 0],
                     [0, 1, 0.167489*np.cos(theta)],
                     [np.cos(theta**2), -1.52003*np.sin(theta), 1],
    ])

# Measurement noise covariance matrix R. Basically what we did for P before. Little goof 
R = calc_R([arr_cbt, arr_hun])
#print(R)

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
    [0, 1, 0],  # cbt mapping 
    [0, 0, 1],  # hunger mapping 
])

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
omega = np.pi/8
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

# xs now contains state estimates, including core body temperature estimates over time
print(type(xs))
print(np.shape(xs))


xs_reshaped = xs.reshape(14999, 3)

xs_cbt = xs_reshaped[:15000, 0]
ys_cbt = np.arange(len(xs_cbt))
time_series = arr_time[:15000]

# Calculate Standard Deviation of Residuals and MSE
residual_std = np.std(residuals, axis=0)
mse = np.mean(residuals**2, axis=0)

residual_std_cbt = residual_std[:, 0]  # Extract standard deviation for CBT
mse_cbt = mse[:, 0]  # Extract MSE for CBT

print("Standard Deviation of Residuals:", residual_std_cbt)
print("Mean Squared Error (MSE):", mse_cbt)

print(f"time_series[0]: {time_series[0]}")
print(f"time_series[-1]: {time_series[-1]}")

file_path = 'minutes_processed_sleep_analysis.csv'
df = pd.read_csv(file_path)

for i in range(8,18):
    start_time = datetime(2023, 7, i, 0, 0)  # Example start time: 05:00
    end_time = datetime(2023, 7, i+1, 0, 0)

    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)

    
    indices = [i for i, t in enumerate(time_series) if start_time <= t <= end_time]
    time_series_df = pd.Series(time_series)
    time_series_filtered = time_series_df.iloc[indices]
    time_series_filtered = time_series_filtered.tolist()

    xs_cbt_filtered = xs_cbt[indices]

    #print(xs_cbt_filtered)
    print(np.shape(xs_cbt_filtered))
    print(type(xs_cbt_filtered))

    #print(time_series_filtered)
    print(np.shape(time_series_filtered))
    print(type(time_series_filtered))
    print()
    print()

    locator = mdates.HourLocator(interval=2)  # ticks every 2 hours
    plt.gca().xaxis.set_major_locator(locator)
    formatter = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().set_xlim(start_time, end_time)

    plt.xticks(rotation=45)
    subset_data = df.iloc[0:1440]

    print(subset_data)
    plt.plot(time_series_filtered, subset_data['value'], label='Sleep Analysis', alpha=0.7, linestyle='-', color='red')
    
    plt.plot(time_series_filtered, xs_cbt_filtered, label='Predicted SCN Activity', color = "blue")
    plt.title('Predicted SCN Activity over 24 Hours')
    plt.xlabel('Time (Hours)')
    plt.ylabel('SCN Activity Estimate')
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(script_dir,'ensemble_plot_pngs', f"ensemble_output_{start_time}_to_{end_time}_.png"))
    plt.clf()
    plt.cla()

plt.plot(ys_cbt, xs_cbt, label='Predicted Ensemble Levels')
plt.title('Predicted Ensemble over Time')
plt.xlabel('Time')
plt.ylabel('Ensemble Estimate')
plt.legend()


# Show the final plot with both graphs
plt.tight_layout()
plt.savefig('ensemble_output.png')
plt.show()