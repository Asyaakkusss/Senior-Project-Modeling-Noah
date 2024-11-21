import numpy as np
import pandas as pd

# Path to the predictions_cbt.csv file
file_path = 'path/to/predictions_cbt.csv'  # Update with actual path

# Load the CBT predictions from the CSV file
# Assuming the first column contains the CBT estimates
cbt_predictions = pd.read_csv(file_path, header=None)  # No header

# Convert to numpy array
cbt_predictions_array = cbt_predictions.to_numpy().flatten()

# Calculate the variance of the CBT predictions
cbt_variance = np.var(cbt_predictions_array)

# Print the variance
print(f'Variance of CBT predictions: {cbt_variance}')
