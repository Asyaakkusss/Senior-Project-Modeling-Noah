import sys 
sys.path.append("/home/asyaakkus/Senior-Project-Modeling-Noah/SleepCycle/")
import data_processing
from data_processing import process_categorical_data, process_numerical_data, calc_R
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


# Load CSV into a DataFrame
df_cbt = pd.read_csv("cbtarray.csv")
df_hun = pd.read_csv("hungerarray.csv")

# Convert to a NumPy array
arr_cbt = df_cbt.to_numpy().flatten()
arr_hun = df_hun.to_numpy().flatten()

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


# Calculate Standard Deviation of Residuals and MSE
residual_std = np.std(residuals, axis=0)
mse = np.mean(residuals**2, axis=0)

residual_std_cbt = residual_std[:, 0]  # Extract standard deviation for CBT
mse_cbt = mse[:, 0]  # Extract MSE for CBT

print("Standard Deviation of Residuals:", residual_std_cbt)
print("Mean Squared Error (MSE):", mse_cbt)

plt.plot(ys_cbt, xs_cbt, label='Predicted Ensemble Levels')
plt.title('Predicted Ensemble over Time')
plt.xlabel('Time Steps')
plt.ylabel('Ensemble Estimate')
plt.legend()


# Show the final plot with both graphs
plt.tight_layout()
plt.savefig('ensemble_output.png')
plt.show()

