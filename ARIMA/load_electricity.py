import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Download: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
file_path = './LD2011_2014.txt'  # Update the file name if needed

# Read the dataset using Pandas 
data = pd.read_csv(file_path, sep=';', header=None)

'''
Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.
Each column represent one client. Some clients were created after 2011. In these cases consumption were considered zero.
All time labels report to Portuguese hour. However all days present 96 measures (24*4). 
Every year in March time change day (which has only 23 hours) the values between 1:00 am and 2:00 am are zero for all points. 
Every year in October time change day (which has 25 hours) the values between 1:00 am and 2:00 am aggregate the consumption of two hours.
'''

'''
Data shape: [ 140257 x 371 ]
            1st col = index of time
            1st row = index of person
            Contain 140256 time x 370 person data in total

'''
#%% Plot one person usage

person_id = 55

# Accessing a single col by index, convert ',' to '.' (float number)
column_data = data.iloc[1:, person_id]

column_data_val = np.array([float(str(element).replace(',', '.')) for element in column_data])


# Plotting a column
plt.plot(column_data_val[-2000:-1000])
plt.xlabel('X in 15min')
plt.ylabel('Y in kW')
plt.title('Consumption')
plt.show()



#%% Get and save all data

data_all = np.zeros([140256, 370])

for i in range(1, 371):
    column_data = data.iloc[1:, i]
    column_data_val = np.array([float(str(element).replace(',', '.')) for element in column_data])
    data_all[:, i-1] = column_data_val

save_name = 'dataset_elec.npy'
np.save(save_name, data_all)



#%% Load and plot the same person

data_my = np.load(save_name)

# Plotting a column
plt.plot(data_my[:, person_id-1])
plt.xlabel('X every 15min')
plt.ylabel('Y in kW')
plt.title('Consumption')
plt.show()
