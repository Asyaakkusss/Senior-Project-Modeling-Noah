import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the hunger data with peaks
times = ["09:00", "13:00", "16:00"]
date = "2024-09-17"
peaks = [1, 1, 1]  # Hunger peaks
sharp_drop = 0.2  # Sharp drop after meals

# Create a pandas DataFrame for minute-by-minute data
start_time = "08:00"  # One hour before the first peak
end_time = "18:00"    # Two hours after the last peak
minute_range = pd.date_range(f"{date} {start_time}", f"{date} {end_time}", freq="T")
hunger_df = pd.DataFrame({"Time": minute_range})

# Interpolation function for hunger
def interpolate_hunger(times, peaks, minute_range, sharp_drop):
    time_minutes = [(pd.Timestamp(f"{date} {t}") - pd.Timestamp(f"{date} {start_time}")).seconds / 60 for t in times]
    hunger_values = np.zeros(len(minute_range))

    for i, peak_time in enumerate(time_minutes):
        # Define intervals for gradual increase and sharp drop
        start_increase = peak_time - 60  # Start ramping up 1 hour before
        end_drop = peak_time + 10  # Hunger drops sharply after 10 minutes

        for j, minute in enumerate(range(len(minute_range))):
            if start_increase <= minute < peak_time:  # Gradual increase
                hunger_values[minute] = (minute - start_increase) / (peak_time - start_increase) * peaks[i]
            elif peak_time <= minute <= end_drop:  # Sharp drop
                hunger_values[minute] = peaks[i] - (minute - peak_time) / (end_drop - peak_time) * (peaks[i] - sharp_drop)
            elif minute > end_drop and hunger_values[minute] < sharp_drop:  # Baseline hunger
                hunger_values[minute] = sharp_drop
    
    return hunger_values

# Add hunger values to the DataFrame
hunger_df["Hunger"] = interpolate_hunger(times, peaks, minute_range, sharp_drop)

# Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(hunger_df["Time"], hunger_df["Hunger"], label="Hunger Level")
plt.xlabel("Time")
plt.ylabel("Hunger")
plt.title("Hunger Levels Throughout the Day")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Save the filled data to a CSV
output_file = "/mnt/data/hunger_data_filled.csv"
hunger_df.to_csv(output_file, index=False)
print(f"Minute-by-minute hunger data saved to {output_file}")
