#!/usr/bin/env python
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
import numpy as np

driver_filename="Data_processing/Data/ContinueBack/h5files_driver/v2d_mframe_00001.h5"
witness_filename="Data_processing/Data/ContinueBack/h5files_witness/v2d_mframe_00001.h5"
no_CB_filename="Data_processing/Data/ContinueBack/h5files_witness/v2d_mframe_00001.h5"

driver_file = h5py.File(driver_filename, 'r')
witness_file = h5py.File(witness_filename, 'r')
no_CB_file = h5py.File(no_CB_filename, 'r')


X_driver=driver_file["X"]
Y_driver=driver_file["Y"]
X_witness=witness_file["X"]
Y_witness=witness_file["Y"]
X_no_CB=no_CB_file["X"]
Y_no_CB=no_CB_file["Y"]

#NonUniformImage assumes the axes give the positions of the cell centres.
#The plot range should therefore extend an additional half cell in each direction
#If the axes instead give the lower edge, X and Y need to be updated.
Xmin=X_driver[0]+0.5*(X_driver[0]-X_driver[1])
Xmax=X_driver[-1]+0.5*(X_driver[-1]-X_driver[-2])
Ymin=Y_driver[0]+0.5*(Y_driver[0]-Y_driver[1])
Ymax=Y_driver[-1]+0.5*(Y_driver[-1]-Y_driver[-2])

#Get E-field
Driver_E_y_index = np.argmin(np.abs(Y_driver))  # Finds the index of the closest value to y = 0
Driver_Ex_field = driver_file["ex"][:, Driver_E_y_index]
Driver_X_array=X_driver[:]

Witness_E_y_index = np.argmin(np.abs(Y_witness))  # Finds the index of the closest value to y = 0
Witness_Ex_field = witness_file["ex"][:, Witness_E_y_index]
Witness_X_array=X_witness[:]

No_CB_E_y_index = np.argmin(np.abs(Y_no_CB))  # Finds the index of the closest value to y = 0
No_CB_Ex_field = no_CB_file["ex"][:, No_CB_E_y_index]
No_CB_X_array=X_no_CB[:]

# Create figure and axis
fig, ax = plt.subplots()
ax.plot(Driver_X_array, Driver_Ex_field, color='tab:blue', label='Driver only')
ax.plot(Witness_X_array, Witness_Ex_field, color='tab:red', label='Witness only')
ax.plot(No_CB_X_array, No_CB_Ex_field, color='tab:red', label='No ContinueBack')

# Labels and title
ax.set_xlabel('x*k_p')
ax.set_ylabel('E_x/E_wb')
ax.set_title('E field comparison using ContinueBack')
ax.legend()

# Show plot
plt.show()


driver_file.close()