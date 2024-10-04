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



#extract VO_2 data 
with open('/home/asyaakkus/Senior-Project-Modeling-Noah/VO2Max.csv', 'r') as file: 
    reader = csv.DictReader(file)

    column_data = [row[col_to_extract] for row in reader]

vo2_rate = np.array(column_data)



#convert the arrays to integers 
def convert_to_integer(array): 
    return np.array(array).astype(float)

RR = convert_to_integer(respiratory_rate)
VO = convert_to_integer(vo2_rate)
HR = convert_to_integer(heart_rate)

print(np.shape(RR), np.shape(VO), np.shape(HR))

print(VO)
plt.plot(RR)
plt.show()

# unified_array = np.array([[RR], [VO], [HR]])
# P = np.cov(unified_array)

# print(P)