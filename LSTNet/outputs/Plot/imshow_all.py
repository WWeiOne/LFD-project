import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


truth = np.load("truth5.npy",allow_pickle=True)
pred = np.load("pred5joint.npy",allow_pickle=True)
pred_non = np.load("pred5.npy",allow_pickle=True)



plt.figure()  # Adjust the figure size if needed
plt.plot([i for i in range(0,2000)], [i for i in range(0,2000)], label='Train', color='blue')  # Plotting list a with blue color

plt.plot([i for i in range(2000,4000)], truth, label='Truth', color='red')  # Plotting list a with blue color
plt.plot([i for i in range(2000,4000)], pred, label='Joint linear&Non-linear model Prediction', color='green')   # Plotting list b with red color
plt.plot([i for i in range(2000,4000)], pred_non, label='Non-linear model Prediction', color='yellow')   # Plotting list b with red color


plt.legend()  # Show legend with labels
plt.xlabel('T - time')  # Label for the x-axis
plt.ylabel('Y - value')  # Label for the y-axis
plt.title('Comparison of joint & Non-linear model')  # Title of the plot
plt.grid(True)  # Show grid
plt.show()  # Display the plot