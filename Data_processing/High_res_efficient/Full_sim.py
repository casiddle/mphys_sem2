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
    Xmin=X[0]+0.5*(X[0]-X[1])
    Xmax=X[-1]+0.5*(X[-1]-X[-2])
    Ymin=Y[0]+0.5*(Y[0]-Y[1])
    Ymax=Y[-1]+0.5*(Y[-1]-Y[-2])

    #Get E-field
    E_y_index = np.argmin(np.abs(Y))  # Finds the index of the closest value to y = 0
    E_x_field = driver_file["ex"][:, E_y_index]
    X_array=X[:]
    return X_array, E_x_field, file["phase"]


driver_filename="Data_processing/Data/High_res_efficient/Driver/v2d_mframe_00006.h5"
witness_filename="Data_processing/Data/High_res_efficient/Witness/v2d_mframe_00006.h5"
initial_Witness_filename="Data_processing/Data/High_res_efficient/Witness/v2d_mframe_00001.h5"
Initial_driver_file="Data_processing/Data/High_res_efficient/Driver/v2d_mframe_00001.h5"

driver_file = h5py.File(driver_filename, 'r')
witness_file = h5py.File(witness_filename, 'r')
no_CB_file = h5py.File(initial_Witness_filename, 'r')
for key in driver_file.keys():
    print(key)


X_driver=np.transpose(driver_file["X"])
Y_driver=np.transpose(driver_file["Y"])
final_driver_phase = driver_file["phase"]
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

Initial_Driver_x_array, Initial_Driver_E_field, initial_driver_phase = get_E_field(Initial_driver_file)

# Create figure and axis
fig, ax = plt.subplots()
#ax.plot(No_CB_X_array, No_CB_Ex_field, color='tab:green', label='Initial Witness only')
ax.plot(Driver_X_array, Driver_Ex_field, color='black', label='Final Driver only')
#ax.plot(Witness_X_array, Witness_Ex_field, color='magenta', label='Final Witness only')
ax.plot(Initial_Driver_x_array, Initial_Driver_E_field, color='orange', label='Initial Driver only')


# Labels and title
ax.set_xlabel('x*k_p')
ax.set_ylabel('E_x/E_wb')
ax.set_title('E comparison with ContinueBack and TreperPoints')
ax.legend()



print("Initial Driver Phase:" + str(initial_driver_phase[0]))
print("Final Driver Phase: " + str(final_driver_phase[0]))
# Show plot
plt.show()


driver_file.close()