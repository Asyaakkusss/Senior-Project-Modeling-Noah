import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.express as px

home_dir = "F:/FALL 2024/Senior-Project-Modeling-Noah"
df = pd.read_csv(os.path.join(home_dir, "HeartRate.csv"))
# Convert 'start' column to datetime
df['start'] = pd.to_datetime(df['start'])

# Sort the dataframe by the 'start' column to ensure correct plotting
df.sort_values('start', inplace=True)

"""
# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df['start'], df['value'], marker='o', linestyle='-', markersize=4)
plt.title('Heart Rate Over Time')
plt.xlabel('Time')
plt.ylabel('Heart Rate (count/min)')
plt.grid(True)
plt.show()
"""

"""
# Plotting a cleaner line graph without markers
plt.figure(figsize=(14, 7))
plt.plot(df['start'], df['value'], linestyle='-')
plt.title('Heart Rate Over Time')
plt.xlabel('Time')
plt.ylabel('Heart Rate (count/min)')
plt.grid(True)
plt.show()
"""

fig = px.line(df, x='start', y='value', title='Interactive Heart Rate Over Time', labels={'start': 'Time', 'value': 'Heart Rate (count/min)'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()