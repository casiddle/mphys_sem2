import sys
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np


matrix_file1=r'Data_processing\Parameters\matrices\S_matrix_1_10.npy'
matrix_file2=r'Data_processing\Parameters\matrices\S_matrix_10.npy'
matrix1=np.load(matrix_file1)
matrix2=np.load(matrix_file2)

print(np.array_equal(matrix1, matrix2))  # Output: True (both are the same)