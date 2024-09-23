import numpy as np
import matplotlib.pyplot as plt 

'''TODO: figure out wtf is going on with the simulated data and why it is just as noisy as the core temp data. Kalman filter is not
Kalmaning'''
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

5. Kalman gain computation....what is Kalman gain? Indicates how much weight should be given to 
new measurements vs the predicted state estimate. Balances uncertain in prediction with uncertainty in measurements. 
High Kalman gain: filter trust measurement more than prediction 
Low Kalman gain: filter trust prediction more than measurement 
In oour case, the measurement is Noah's data (gotten from his apple watch sensor) and the prediction is whatever the 
filter decides the value actually should be at that point in time. Like so: 

k_t = (v_t * c_t) / (c_t**2 * v_t) + 18.88**2

6. computer final estimate of CT using everything that we have so far: 

CT_final = CT_new + k_t(heart_rate - (b2 * CT_new**2 + b1 * CT_new + b0))

CT_final = CT_new + k_t(heart_rate - (-4.5714 * CT_new**2 + 384.4286 * CT_new - 7887.1))

7. compute final varine of final CT estimate (vt): 

v_t = (1 - k_t*c_t)v_t

**please note that the sigma is the SD for binned heart_rate (it is 18.88 plus/minus 3.78 bpm). 

'''




def simulate_heart_rate(): 

    num_samples = 1440  # Number of data points to generate
    mean_hr = 70        # Average heart rate
    std_dev = 5         # Standard deviation of heart rate

    # Generate random heart rate data
    heart_rate = np.random.normal(mean_hr, std_dev, num_samples)

    return heart_rate

def simulate_heart_rate_detailed(): 

    # Parameters
    num_samples = 1440        # Number of data points to generate
    mean_hr = 70              # Average heart rate
    std_dev = 5               # Standard deviation of heart rate
    time = np.arange(0, 1440, 1)  # Simulated time in hours
    print(time)
    # Base heart rate with random noise
    base_hr = np.random.normal(mean_hr, std_dev, num_samples)

    # Add a periodic component to simulate circadian rhythm
    circadian_rhythm = 5 * np.sin(2 * np.pi * time / 24)

    # Add an exercise effect: A peak in heart rate at 3-4 hours
    exercise_effect = np.exp(-(time - 3)**2 / (2 * 0.5**2)) * 20

    # Total heart rate
    heart_rate = base_hr + circadian_rhythm + exercise_effect

    return heart_rate

#array for preliminary estimates of CT
CT_hat = [] 

#array for final estimates of CT
CT_finalboss = [] 

#array for time (in minutes)
time = [] 

#array for variance of preliminary estimate 
v_t = [] 

#array for extended KF mapping function variance coefficient 
c_t = [] 

#array for the variance of the final boss estimate
v_t_finalboss = [] 

#array for kalman gain over time 
k_t = []

#array for heart rate 
heart_rate = simulate_heart_rate() 

'''instantiate the first iteration of this shit:''' 

CT_hat.append(37) # first value of CT_hat (there is no previous)

v_t.append(0.000484) # first value for variance (since there is no previous)

# equation for calculating c_t for the first iteration 
c_t_firstit = (-9.1428 * CT_hat[0]) + 384.4286
c_t.append(c_t_firstit) #appending the result of the firstit calculation to c_t 

#computing kalman gain for the first iteration 
k_t_firstit = (v_t[0] * c_t[0])/((c_t[0]**2 * v_t[0]) + 18.88**2)
k_t.append(k_t_firstit)

#compute final boss bc I am based and red pilled 
CT_finalboss_firsit = CT_hat[0] + k_t[0]*(heart_rate[0] - (-4.5714 * CT_hat[0]**2 + 384.4286 * CT_hat[0] - 7887.1))
CT_finalboss.append(CT_finalboss_firsit)

#compute vt of the ifnal boss 
v_t_finalboss.append((1- k_t[0]*c_t[0])*v_t[0]) 

'''keep going'''

for i in range(1, 1440): 
    CT_hat.append(CT_finalboss[i-1])

    v_t.append(v_t[i-1] + 0.000484) 

    c_t_it = (-9.1428 * CT_hat[i]) + 384.4286
    c_t.append(c_t_it) 

    #computing kalman gain for the first iteration 
    k_t_it = (v_t[i] * c_t[i])/((c_t[i]**2 * v_t[i]) + 18.88**2)
    k_t.append(k_t_it)

    CT_finalboss_it = CT_hat[i] + k_t[i]*(heart_rate[i] - (-4.5714 * CT_hat[i]**2 + 384.4286 * CT_hat[i] - 7887.1))
    CT_finalboss.append(CT_finalboss_it)

    v_t_finalboss.append(1 - k_t[i]*c_t[i]*v_t[i])



iterations = np.arange(0, 1440)

plt.subplot(2, 2, 1)
plt.plot(iterations, heart_rate)
plt.title('Simulated Heart Rate Data (BPM)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Heart Rate (BPM)')

plt.subplot(2, 2, 2)
plt.title('Core Temperature Data from Kalman Filtering')
plt.xlabel("Time (Minutes)")
plt.ylabel("Core Temperature (C)")
plt.plot(iterations, CT_finalboss)

plt.show()