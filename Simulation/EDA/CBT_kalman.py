import matplotlib.pyplot as plt
import pandas as pd
import os
import plotly.express as px

home_dir = "/Users/monugoel/Desktop/CSDS_395/"
df = pd.read_csv(os.path.join(home_dir, "predictions_cbt.csv"), header = None)

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

fig = px.line(df, x=df.index, y=0, title='CBT Kalman Filtered', labels={'start': 'Time', 'value': 'Temperature (F)'})
fig.update_xaxes(rangeslider_visible=True)
fig.show()