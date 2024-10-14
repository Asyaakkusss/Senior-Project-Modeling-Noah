from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'HKCategoryTypeIdentifierSleepAnalysis.csv'


#two things must change: how the interval is chosen, and what values go into the data


def aggregate_data_by_interval(file, interval):
    # Convert 'start' to datetime
    df = pd.read_csv(file)
    df['start'] = pd.to_datetime(df['start'], errors='coerce')

    # Drop rows where 'start' could not be converted to datetime
    df = df.dropna(subset=['start'])

    # Check if it's a sleep data file based on the presence of 'end' column
    is_sleep_data = interval == 'sleep' 
    
    # For sleep data, convert 'end' to datetime
    if is_sleep_data:
        df['end'] = pd.to_datetime(df['end'], errors='coerce')
        df = df.dropna(subset=['end'])  # Drop rows with invalid end dates

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
    elif interval == 'sleep':
        time_delta = timedelta(hours=8)
    else:
        raise ValueError("Interval must be 'minutes', 'hours', 'days', or 'sleep'")

    # Create an empty list to store results
    results = []
    time_counter = 0
    current_time = start_time

    if interval == 'sleep' and is_sleep_data:
        # Sleep-specific aggregation
        sleep_start = None
        sleep_end = None

        for index, row in df.iterrows():
            if sleep_start is None:
                sleep_start = row['start']
                sleep_end = row['end']
            else:
                # Check for more than 8-hour gap, to separate sleep spans
                if row['start'] - sleep_end > time_delta:
                    results.append({
                        'source': row['source'],  # Source from current row
                        'time': sleep_start,
                        'end': sleep_end,
                        'value': (sleep_end - sleep_start).total_seconds() / 3600,  # Duration in hours
                    })
                    sleep_start = row['start']
                    sleep_end = row['end']
                else:
                    sleep_end = row['end']

        # Add final sleep span if exists
        if sleep_start is not None:
            results.append({
                'source': df['source'].iloc[-1],
                'time': sleep_start,
                'end': sleep_end,
                'value': (sleep_end - sleep_start).total_seconds() / 3600,
            })
    
    else:
        # General aggregation for non-sleep data files
        while current_time <= df['start'].max():
            interval_data = df[(df['start'] >= current_time) & (df['start'] < current_time + time_delta)]
            
            if not interval_data.empty:
                mean_value = interval_data['value'].mean()
                results.append({
                    'source': interval_data['source'].iloc[0],
                    'time': time_counter,
                    'value': mean_value,
                    'unit': interval_data['unit'].iloc[0],
                })

            # Move to the next interval
            current_time += time_delta
            time_counter += 1

    # Convert results into a DataFrame
    result_df = pd.DataFrame(results)
    return result_df


# Get current axis
ax = plt.gca()

aggregated_df = aggregate_data_by_interval(file_path, interval = 'sleep')
aggregated_df.plot(kind='line', x='time', y='value')
plt.show()


# Saving the result to a new CSV file
aggregated_df.to_csv('PureSleepTime.csv', index=False)
