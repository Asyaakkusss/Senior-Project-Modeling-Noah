'''
Im going to need a state vector like so: [CBT HR ST]

we need parameters describing how skin temp and heart rate impact core body temp 

Process model: 
A = 
[1 alpha beta]
[0 1     0]
[0 0     1]

x_t+1 = A*xt + w_t

Measurement model: H matrix with heart rate and skin temperature from state vector 

H = [0 1 0]
    [0 0 1]

Q covariance (for process model): process noise 
[Q_CBT 0 0]
[0 Q_HR 0]
[0 0 Q_ST]

R covariance (for measurement model): variances for heart rate and skin temp 
R = [R_HR 0]
    [0 R_ST]


'''
import numpy as np 
import csv 
import matplotlib.pyplot as plt 
col_to_extract = "value"
import pandas as pd 

#extract respiratory rate data

#extract heart rate data 
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/HeartRate.csv', 'r') as file:
    reader = csv.DictReader(file)
    
    column_data = [row[col_to_extract] for row in reader]

heart_rate = np.array(column_data)


#extract respiratory rate data 
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/RespiratoryRate.csv', 'r') as file: 
    reader = csv.DictReader(file)

    column_data = [row[col_to_extract] for row in reader]

respiratory_rate = np.array(column_data)


#extract basal energy burned data 
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/BasalEnergyBurned.csv', 'r') as file: 
    reader = csv.DictReader(file)

    column_data = [row[col_to_extract] for row in reader]

basal_energy_burned = np.array(column_data)



#convert the arrays to integers 
def convert_to_integer(array): 
    return np.array(array).astype(float)

RR = convert_to_integer(respiratory_rate)
BE = convert_to_integer(basal_energy_burned)
HR = convert_to_integer(heart_rate)

print(np.shape(RR), np.shape(BE), np.shape(HR))

'''
we find the P matrix by taking the variance of each of the arrays at consistent time stamps. for now, we will just focus on 
respiratory rate and heart rate, since I am unsure whether or not VO is salvageable data point. There is not enough data 
and it is too spread out. I am going to use basal energy burned instead. 
'''

'''
We need to truncate the BE and HR arrays to be the same size as RR. With HR, it should be straightforward because both the 
measurements from the watch are taken in count/min. 

The timestamps don't align and this is a big problem. We have to probably use the interpolate method in pandas to make this 
work at all. 


'''

#data processing 
df = pd.read_csv("/home/asyaakkus/Senior-Project-Modeling-Noah/RespiratoryRate.csv")

df['start'] = pd.to_datetime(df['start'])
df.set_index('start', inplace=True)

common_time = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1T')

respir_interpolated = df['value'].reindex(common_time).interpolate()

aligned_rr_df = pd.DataFrame({
    'start': common_time, 
    'value': respir_interpolated
})

print(aligned_rr_df)
#unified_array = np.array([[RR], [BE], [HR]])
#P = np.cov(unified_array)
