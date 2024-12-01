import numpy as np 
import sys 
sys.path.append("./SleepCycle")
from data_processing import process_categorical_data, process_numerical_data

#preprocess the data 
resp_rate_csv_string = "data/RespiratoryRate.csv"
heart_rate_csv_string = "data/HeartRate.csv"
basal_rate_csv_string = "data/BasalEnergyBurned.csv"
col_interest = 'start'
processed_respiratory = process_numerical_data(resp_rate_csv_string, col_interest)
processed_heart_rate = process_numerical_data(heart_rate_csv_string, col_interest)
processed_basal_rate = process_numerical_data(basal_rate_csv_string, col_interest)

#the three arrays have null values, so we crop the nulls out to leave as much valid data as possible  
processed_respiratory = processed_respiratory[612:]
processed_heart_rate = processed_heart_rate[612:]
processed_basal_rate = processed_basal_rate[612:]

#thr three arrays are now different lengths, so we find the minimum length and cut off the maximum index of an array at that minimum length 
min_length = min(len(processed_respiratory), len(processed_heart_rate), len(processed_basal_rate))

#create the index values for the p matrix by finding covariance between the three arrays 
processed_basal_rate = processed_basal_rate[:min_length]
processed_heart_rate = processed_heart_rate[:min_length]
processed_respiratory = processed_respiratory[:min_length]

processed_heart_rate = processed_heart_rate - np.mean(processed_heart_rate)

time = np.arange(0, len(processed_basal_rate))

sampling_rate = 1

window = np.hamming(len(processed_heart_rate))  # Use a Hamming window
processed_heart_rate = processed_heart_rate * window
fft_result = np.fft.fft(processed_heart_rate)

frequencies = np.fft.fftfreq(len(processed_heart_rate), d=1/sampling_rate)

print(frequencies)
amplitudes = np.abs(fft_result)  # Amplitude spectrum
phases = np.angle(fft_result)   # Phase spectrum

print(amplitudes)

# Set a threshold for significant components
threshold = 0.1 * np.max(amplitudes)  # Keep components above 10% of the max amplitude
dominant_indices = np.where(amplitudes > threshold)

dominant_frequencies = frequencies[dominant_indices]
dominant_amplitudes = amplitudes[dominant_indices]
dominant_phases = phases[dominant_indices]

print(dominant_frequencies, dominant_amplitudes, dominant_phases)

# Time array for reconstruction
duration = len(processed_heart_rate) / sampling_rate
t = np.linspace(0, duration, len(processed_heart_rate), endpoint=False)

# Reconstruct the signal
reconstructed_signal = np.zeros_like(t)
for i in range(len(dominant_indices)):
    f = dominant_frequencies[i]   # Frequency
    A = dominant_amplitudes[i]    # Amplitude
    phi = dominant_phases[i]      # Phase
    reconstructed_signal += A * np.sin(2 * np.pi * f * t + phi)

import matplotlib.pyplot as plt

# Plot the original signal
plt.plot(t, processed_heart_rate, label='Original Signal')

# Plot the reconstructed signal
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig("fourier.png")