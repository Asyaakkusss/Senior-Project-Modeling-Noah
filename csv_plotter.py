import pandas as pd
import matplotlib.pyplot as plt

def plot(file, interval):
    df = pd.read_csv(file)
    ax = plt.gca()
    row = interval
    while row >= df.shape[0]:
        plot_df = df.iloc[:row]
        plot_df.plot(kind='line', x='time', y='value')
        plt.show()

plot("PureHeartRate.csv", 200)
