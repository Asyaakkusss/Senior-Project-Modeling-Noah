import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from numpy.random import randn

# Define constants
dt = 1.0  # time step (1 second)
omega = np.pi / 6  # angular velocity (30 degrees per second)

# Rotation matrix function for angular motion
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

# Initialize the Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=2)
kf.F = rotation_matrix(omega * dt)  # state transition matrix (rotation matrix)
kf.H = np.eye(2)  # measurement matrix (we directly measure position)
kf.R = np.eye(2) * 0.5  # measurement noise covariance
kf.Q = np.eye(2) * 0.01  # process noise covariance
kf.x = np.array([1, 0])  # initial state (start on the unit circle at (1,0))
kf.P = np.eye(2)  # initial uncertainty covariance

# Simulate noisy measurements of circular motion
num_steps = 50
true_positions = []
measurements = []
predictions = []

for i in range(num_steps):
    # True position is based on the rotation matrix
    true_position = rotation_matrix(omega * i * dt).dot(np.array([1, 0]))
    true_positions.append(true_position)
    
    # Noisy measurements
    measurement = true_position + randn(2) * 0.5
    measurements.append(measurement)
    
    # Kalman filter prediction and update
    kf.predict()  # predict next state
    kf.update(measurement)  # update with noisy measurement
    predictions.append(kf.x)  # store the prediction

# Convert results to numpy arrays for easy plotting
true_positions = np.array(true_positions)
measurements = np.array(measurements)
predictions = np.array(predictions)

# Plot the results
plt.figure(figsize=(8, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], label='True Path', color='blue', linewidth=2)
plt.scatter(measurements[:, 0], measurements[:, 1], label='Noisy Measurements', color='red', marker='x')
plt.plot(predictions[:, 0], predictions[:, 1], label='Kalman Filter Estimate', color='green', linestyle='--', linewidth=2)
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Kalman Filter Tracking Circular Motion')
plt.grid(True)
plt.show()
