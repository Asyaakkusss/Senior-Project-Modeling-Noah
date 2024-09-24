import pandas as pd
import matplotlib.pyplot as plt

#interval is set to 1440 as that is the number of minutes in a day
#stop is in case you want to stop plotting after a specific time
#begin is if you want to start later in the file
def plot(file, interval=1440, stop=None, begin=0):
    df = pd.read_csv(file)

    if 'time' not in df.columns or 'value' not in df.columns:
        raise ValueError("The CSV must contain 'time' and 'value' columns.")
    
    prev_row = begin
    row = prev_row + interval
    if not stop:
        stop = df.shape[0]
    while row <= stop:
        plot_df = df.iloc[prev_row:row]
        
        # Plot the data
        plot_df.plot(kind='line', x='time', y='value', ax=plt.gca())
        plt.title(f"Data from {prev_row} to {row}")
        plt.show()
        
        # Increase the row limit for the next plot
        prev_row = row
        row += interval

#Change the data as necessary
plot('PureHeartRate.csv', stop=10000)
