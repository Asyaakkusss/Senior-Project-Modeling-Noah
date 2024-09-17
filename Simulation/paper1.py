import numpy as np

# Define constants
a1 = 1
a0 = 0
a2 = 1
gamma_squared = 0.022**2
b1 = 384.4286
b2 = -4.5714
b0 = -7382.3
sigma_squared = 18.882
c2 = 1  # Set a default value for c2 if not provided

# Initialize previous CT and variance
CTt_minus_1 = 0  # Previous CT estimate, set initial value as needed
vt_minus_1 = 0  # Previous variance, set initial value as needed

# Define the observation (this would be provided in a real scenario)
HRt = 0  # Example observation, set this value as needed

# Step 1: Compute the preliminary estimate of CT
CTt_hat = a1 * CTt_minus_1 + a0

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
