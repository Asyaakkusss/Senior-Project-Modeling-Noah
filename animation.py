import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pandas as pd
from datetime import datetime

home_dir = "/Users/monugoel/Desktop/CSDS_395/"

hunger_y_data = pd.read_csv(os.path.join(home_dir, "Ensemble/hungerarray.csv"), header = None)[0].values

date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
time_values = pd.read_csv(os.path.join(home_dir, "Ensemble/timearray.csv"), parse_dates=["Datetime"])['Datetime'].values


start_time = datetime(2024, 2, 13, 0, 0)  # Example start time: 05:00
end_time = datetime(2024, 2, 14, 0, 0)
start_time = np.datetime64(start_time).astype('datetime64[ns]')
end_time = np.datetime64(end_time).astype('datetime64[ns]')

print(f"Start time: {start_time}, type: {type(start_time)}")
print(f"End time: {end_time}, type: {type(end_time)}")
print(f"Time values: {time_values}, type: {time_values.dtype}")

print(f"Min time in time_values: {time_values.min()}")
print(f"Max time in time_values: {time_values.max()}")

# indices = []
# for time in time_values:
#     if (start_time <= time) and (time <= end_time):
#         print(f"flag raised")
# print(indices)

time_value_filtered = time_values[7962:9401]
hunger_y_data_filtered = hunger_y_data[7962:9401]

y_data = hunger_y_data_filtered
y_min = min(y_data)-1
y_max = max(y_data)+1
x_data = time_value_filtered

current_x = []
current_y = []

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2) 

ax.set_xlim(x_data[0], x_data[-1])  # Full x-axis range
ax.set_ylim(y_min, y_max)
ax.tick_params(axis='x', rotation=45)
ax.set_title("Subject's Hunger Level over 24 Hours")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global current_x, current_y

    # Append 5 points to the current data
    start = frame * 5
    end = start + 5
    current_x.extend(x_data[start:end])
    current_y.extend(y_data[start:end])
    print(len(current_x))
    print(len(current_y))
    print(current_x[-1])

    # Update the line data
    line.set_data(current_x, current_y)

    if frame == num_frames - 1:
        current_x.clear()
        current_y.clear()
        #ax.clear()
        #ax.set_xlim(x_data[0], x_data[-1])  # Reset x-axis
        #ax.set_ylim(y_min, y_max)          # Reset y-axis
       #ax.tick_params(axis='x', rotation=45)

    return line,

num_frames = len(x_data) // 5  # Number of updates
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)

plt.show()
