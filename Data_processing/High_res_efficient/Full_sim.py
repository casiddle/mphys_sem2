#!/usr/bin/env python
import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
import numpy as np


def get_E_field(file_name):
    file = h5py.File(file_name, 'r')
    X = file["X"]
    Y = file["Y"]
    #Get E-field
    E_y_index = np.argmin(np.abs(Y))  # Finds the index of the closest value to y = 0
    E_x_field = file["ex"][:, E_y_index]
    X_array=X[:]
    file.close()
    return X_array, E_x_field


final_driver_file="Data_processing/Data/High_res_efficient/Driver/v2d_mframe_00005.h5"
final_witness_file="Data_processing/Data/High_res_efficient/Witness/v2d_mframe_00009.h5"
initial_witness_file="Data_processing/Data/High_res_efficient/Witness/v2d_mframe_00001.h5"
initial_driver_file="Data_processing/Data/High_res_efficient/Driver/v2d_mframe_00001.h5"

Initial_Driver_x_array, Initial_Driver_E_field = get_E_field(initial_driver_file)
Final_Driver_x_array, Final_Driver_E_field = get_E_field(final_driver_file)
Initial_Witness_x_array, Initial_Witness_E_field = get_E_field(initial_witness_file)
Final_Witness_x_array, Final_Witness_E_field = get_E_field(final_witness_file)

# Create figure and axis
fig, ax = plt.subplots()
ax.plot(Initial_Witness_x_array, Initial_Witness_E_field, color='tab:green', label='Initial Witness only')
ax.plot(Initial_Driver_x_array, Initial_Driver_E_field, color='black', label='Initial Driver only')
ax.plot(Final_Witness_x_array, Final_Witness_E_field, color='magenta', label='Final Witness only')
ax.plot(Final_Driver_x_array, Final_Driver_E_field, color='orange', label='Final Driver only')
plt.legend()
plt.show()

