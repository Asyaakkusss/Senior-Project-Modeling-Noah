from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/Users/noahh/Documents/GitHub/Senior-Project-Modeling-Noah/HeartRate.csv'


def aggregate_data_by_interval(file, interval):
    # Convert 'start' to datetime
    df = pd.read_csv(file)
    df['start'] = pd.to_datetime(df['start'], errors='coerce')

    # Drop rows where 'start' could not be converted to datetime
    df = df.dropna(subset=['start'])
    
    # Sort by 'start' time to ensure chronological order
    df = df.sort_values(by='start')

    # Set the start time as the base reference
    start_time = df['start'].min()

    # Create a new time index based on the given interval
    if interval == 'minutes':
        time_delta = timedelta(minutes=1)
    elif interval == 'hours':
        time_delta = timedelta(hours=1)
    elif interval == 'days':
        time_delta = timedelta(days=1)
    else:
        raise ValueError("Interval must be 'minutes', 'hours', or 'days'")

    # Create an empty list to store results
    results = []
    time_counter = 0
    current_time = start_time

    while current_time <= df['start'].max():
        # Filter the data for the current interval
        interval_data = df[(df['start'] >= current_time) & (df['start'] < current_time + time_delta)]

        if not interval_data.empty:
            # Calculate the mean of 'value' for the interval
            mean_value = interval_data['value'].mean()

            # Add the aggregated result to the results list
            results.append({
                'source': interval_data['source'].iloc[0],  # Take the source from the first row
                'time': time_counter,
                'value': mean_value,
                'unit': interval_data['unit'].iloc[0],  # Take the unit from the first row
            })

        # Move to the next interval
        current_time += time_delta
        time_counter += 1

    # Convert results into a DataFrame
    result_df = pd.DataFrame(results)
    return result_df


# Get current axis
ax = plt.gca()

aggregated_df = aggregate_data_by_interval(file_path, 'minutes')
aggregated_df.plot(kind='line', x='time', y='value')
plt.show()


# Saving the result to a new CSV file
aggregated_df.to_csv('PureHeartRate.csv', index=False)
