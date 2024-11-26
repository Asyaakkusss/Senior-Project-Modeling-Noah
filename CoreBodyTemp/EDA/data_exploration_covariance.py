import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
heart_rate_path = 'data\HeartRate.csv'
basal_energy_path = 'data\BasalEnergyBurned.csv'
respiratory_rate_path = 'data\RespiratoryRate.csv'

# Read the CSV files
heart_rate_data = pd.read_csv(heart_rate_path)
basal_energy_data = pd.read_csv(basal_energy_path)
respiratory_rate_data = pd.read_csv(respiratory_rate_path)

# Extract the relevant 'value' columns
hr_values = heart_rate_data['value']
bmr_values = basal_energy_data['value']
rr_values = respiratory_rate_data['value']

# Combine the datasets into a single DataFrame
combined_data = pd.DataFrame({
    "Heart Rate (HR)": hr_values,
    "Basal Energy Burned (BMR)": bmr_values,
    "Respiratory Rate (RR)": rr_values
}).dropna()  # Drop rows with missing values

# Compute the covariance and correlation matrices
covariance_matrix = combined_data.cov()
correlation_matrix = combined_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(combined_data.columns)), combined_data.columns, rotation=45)
plt.yticks(range(len(combined_data.columns)), combined_data.columns)
plt.title('Correlation Matrix')
plt.show()

print(covariance_matrix)
print(correlation_matrix)


