import pandas as pd
import numpy as np

def transform_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure the 'start' column is in datetime format and set it as the index
    input_df['start'] = pd.to_datetime(input_df['start'], errors='coerce')
    input_df.set_index('start', inplace=True)

    # Drop rows with invalid 'value'
    input_df['value'] = pd.to_numeric(input_df['value'], errors='coerce')
    input_df = input_df.dropna(subset=['value'])
    
    '''
    # Find start and end times of the input DataFrame
    start_time = input_df.index.min()
    end_time = input_df.index.max()
    '''
    start_time = pd.Timestamp('2023-07-07 01:08:27-0400')
    end_time = pd.Timestamp('2024-09-05 08:27:27-0400')
    print(f"Start Time: {start_time}, End Time: {end_time}")

    # Create a new DataFrame with 1-minute intervals as index
    new_index = pd.date_range(start=start_time, end=end_time, freq='min')
    new_df = pd.DataFrame(index=new_index, columns=['value'])
    new_df['value'] = 0

    print(new_df.head())
    print(input_df.head())

    # Populate the new DataFrame
    for new_time in new_index:
        # Adjust the time range to stay within bounds
        start_range = new_time - pd.Timedelta(hours=1)
        end_range = new_time + pd.Timedelta(hours=1)

        # Filter the input DataFrame for the given range
        time_window = input_df.loc[(input_df.index >= start_range) & (input_df.index <= end_range)]

        if not time_window.empty:
            # Find the closest value by time difference
            closest_index = np.abs(time_window.index - new_time).argmin()
            closest_value = time_window.iloc[closest_index]['value']
            new_df.at[new_time, 'value'] = closest_value

    # Save the processed data to a CSV file
    new_df.to_csv("processed_sleep_analysis.csv", index=False)

    return new_df
