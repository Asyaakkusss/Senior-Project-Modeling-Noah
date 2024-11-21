'''I am going to now simulate this paper that brings heart rate into the Kalman filtering model: 
https://arc.aiaa.org/doi/epdf/10.2514/6.2022-1271 

The math of this paper goes like so: 

We have a basic time update model: 
C_k_hat = C_(k -1) + wkc
C is the core temperature, subscript k is the time step, and W^c is zero mean Gaussian noise with
variance sigma squared. 

There is uncertainty in the core temperature prediction, of course. It is denoted by: 
p_hat_k = p_(k - 1) + sigma**2c

In the update step, a measurement of heart rate or skin temp is provided and the following observation 
model is used to predict the measurement: 
y_hat_k = b_1 * C_k + b0 + w_k_y
y is a heart rate measurement H or a skin temp measurement S. w_k_y is random noise. b1 and b0 are obtained from linear 
regression. 

Kalman gain is calculted like so: 
K_k = (p_k * b1)/(b1 * p_k + sigma**2)


Algorithm: 

Inputs: 
1. Core temp, Heart Rate, Skin Temperature regression coefficients and variances
2. Initial Core temperature
3. Heart rate readings 
4. Skin temp readings 

Outputs: core temperature estimates C_hat

for each time step...
predict core temperature using: C_k_hat = C_(k -1) + wkc and p_hat_k = p_(k - 1) + sigma**2c

'''

import numpy as np
import matplotlib.pyplot as plt 

#simulating heart rate data based on a normal distribution 
def simulate_heart_rate(num_samples=200, mean_hr=70, std_dev=5):
    heart_rate = np.random.normal(mean_hr, std_dev, num_samples)
    return heart_rate

#simulating skin temperature data based on a normal distribution 
def simulate_skin_temp(num_samples=200, mean_st=33.5, std_dev=5):
    skin_temp = np.random.normal(mean_st, std_dev, num_samples)
    return skin_temp

# Parameters
num_samples = 200
C_initial = 37.0  # Initial core temperature estimate
p_initial = 1.0   # Initial variance estimate
sigma_c = 0.2     # Process noise standard deviation (σ_c)
sigma_y = 0.5     # Measurement noise standard deviation (σ_y)
b0_heart = -1858  # Intercept from linear regression (heart rate)
b1_heart = 51.92  # Slope from linear regression (heart rate)

b1_skin = 2.286   # Slope from linear regression (skin temperature)
b0_skin = -50.12  # Intercept from linear regression (skin temperature)

# Simulate heart rate and skin temperature
heart_rate = simulate_heart_rate()
skin_temp = simulate_skin_temp()

# Initialize lists to store core temperature and variance estimates
C_hat = [C_initial]  # Predicted core temperature before update 
p_k_hat = [p_initial]  # Predicted variance before update 
core_temp_finalboss = [C_initial]  # Updated core temperature after measurement
p_k = [p_initial]  # Updated variance after measurement

# Kalman Filter Loop
for i in range(1, num_samples):
    # Prediction Step
    C_hat.append(core_temp_finalboss[-1] + np.random.normal(0, sigma_c))  # C_hat[k]
    p_k_hat.append(p_k[-1] + sigma_c**2)  # p_k_hat[k]

    # Measurement Update Step (alternating between heart rate and skin temp)
    if i % 2 == 0:
        # Heart rate measurement update
        y_k = heart_rate[i]
        y_k_pred = b1_heart * C_hat[i] + b0_heart  # Predicted heart rate
        K_k = p_k_hat[i] * b1_heart / (b1_heart**2 * p_k_hat[i] + sigma_y**2)
        p_k_updated = p_k_hat[i] * (1 - K_k * b1_heart)
        p_k.append(p_k_updated)


    else:
        # Skin temperature measurement update
        y_k = skin_temp[i]
        y_k_pred = b1_skin * C_hat[i] + b0_skin  # Predicted skin temperature
        K_k = p_k_hat[i] * b1_skin / (b1_skin**2 * p_k_hat[i] + sigma_y**2)
        p_k_updated = p_k_hat[i] * (1 - K_k * b1_skin)
        p_k.append(p_k_updated)



    
    # Update Step
    C_updated = C_hat[i] + K_k * (y_k - y_k_pred)
    core_temp_finalboss.append(C_updated)

# Results

iterations = np.arange(0, 200)

plt.plot(iterations, core_temp_finalboss)
plt.show()
plt.plot(iterations, skin_temp)
plt.show()