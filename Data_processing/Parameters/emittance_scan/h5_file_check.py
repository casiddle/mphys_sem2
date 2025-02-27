import sys
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

filepath1=r'Data_processing\Parameters\emittance_scan\emittance-fix_no_CB\h5files\v3d_synchrotron_00001.h5'
filepath2=r'Data_processing\Parameters\emittance_scan\emittance-fix_no_CB\h5files\v3d_synchrotron_00040.h5'

def read_h5_head(file_path):
  # Open the HDF5 file in read mode
  with h5py.File(file_path, "r") as h5file:
      # List all datasets in the file
      names=['Synchrotron3D']
      for name in names:
          dataset = h5file[name]
          if isinstance(dataset, h5py.Dataset):  # Ensure it's a dataset, not a group
              print(f"Dataset: {name}")
              print("First 5 values:\n", dataset[:5])  # Get first 5 values
              print("-" * 40)

read_h5_head(filepath1)
read_h5_head(filepath2)

