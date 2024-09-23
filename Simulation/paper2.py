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
