import numpy as np

# Define constants
a1 = 1
a0 = 0 #time update model intercept
a2 = 1
gamma_squared = 0.022**2

# extended kalman filter coefficients 
b1 = 384.4286 # linear coefficient 
b2 = -4.5714 # quadratic coefficient 
b0 = -7382.3 # intercept 
sigma_squared = 18.882
c2 = 1  # Set a default value for c2 if not provided

# Initialize previous CT and variance
CTt_initial = 37  # Initial Core temperature estimate, based on the average of 37C
vt_initial = 0.25  # Initial variance, based on average standard deviation in most individuals of ~0.5C

# Initial heart rate 
heart_rate = 60  # Example observation, set this value as needed

# Step 1: Compute the preliminary estimate of CT
CTt_hat = a1 * CT_prev + a0

# Step 2: Compute the preliminary estimate of the variance
vt_hat = a2 * vt_minus_1 + gamma_squared

# Step 3: Compute the extended KF mapping function variance coefficient
ct = 2 * b2 * CTt_hat + b1

# Step 4: Compute the Kalman gain
kt = vt_hat * ct / (c2 * ct * vt_hat + sigma_squared)

# Step 5: Compute the final estimate of CT
expected_HR = b2 * CTt_hat**2 + b1 * CTt_hat + b0
CTt = CTt_hat + kt * (HRt - expected_HR)

# Step 6: Compute the variance of the final CT estimate
vt = (1 - kt * ct) * vt_hat

# Print results
print(f"Preliminary CT estimate (ˆCTt): {CTt_hat}")
print(f"Preliminary variance estimate (ˆvt): {vt_hat}")
print(f"Extended KF mapping function variance coefficient (ct): {ct}")
print(f"Kalman gain (kt): {kt}")
print(f"Final CT estimate (CTt): {CTt}")
print(f"Final variance of CT estimate (vt): {vt}")

'''
var1: we first need to create an array that stores the values of CT over time.
CT stands for core body temperature.

var2: we then need to create an array t that has the same amount of indices as 
CT. 

var3: we need to create an array for variance too. It will have the same number of 
indices as CT and t arrays. 

GIVENS: 
current observation of HRt (heart rate). We will simulate heart rate using 
a Gaussian method. 
Previous CT estimate (previous value in our array. nous commencons par zero)
Previous variance (previous value in the variance array. we also commencons par zero)


t will be in minutes. We will apply the 6 equations below iteratively 
for every value of t to estimate CT and the corresponding variance. 

1. We start with a CT_0 value and the initial variance v_0. 

2. We compute our first CT estimate from our base case that was established in 1
and the other brain rot that I have generated in this long comment thus far, like so: 
CT_new = a1 * CT_prev + a0. 

**note: in practice, with variance not involved, CT should not change from time stamp to 
time stamp. This is why the authors say that a1 = 1 and a0 = 0. This brings CT_new = CT_prev. 

3. Computer preliminitary estimate of the variance of the CT estimate computed previously. This 
is how we anticipate the CT to change over time. Like so: 
vt_new = a1**2 + vt_prev + gamma**2
plugging in a1 = 0, we get the following: 
vt_new = vt_prev + 0.000484 

**please note that gamma = 0.022. Because of this, the variance is changing over time. 

4. We calculate the parameter that determines how much confidence the filter has in its predictions. 
c_t = 2*b2*CT_new + b1 = 2 * -4.5714 * CT1 + 384.4286
c_t = -9.1428 * CT_new  + 384.4286 

5. Kalman gain computation....what is Kalman gain? 

'''