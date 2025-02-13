import sys
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.special import kv
from scipy.integrate import quad
from scipy import integrate
from scipy import signal
import time

import numpy as np


def sum_over_theta_and_phi(synchrotron_array):
    photons_per_crit_energy = np.sum(synchrotron_array, axis=(0,1))
    return photons_per_crit_energy

def integrand(x):
    return kv(nu, x)

def S_integral(E, E_c, upper_bound):

    ratio = E / E_c
    array=np.linspace(ratio,upper_bound)
    integrand_array=integrand(array)
    # Perform integration for each scalar value of ratio
    #integral_result = integrate.simpson(integrand_array, x=array)
    integral_result, _ = quad(integrand, ratio, upper_bound)
    result = ratio * integral_result  # Multiply by ratio
    return result


def S_integral2(E, E_c, upper_bound):
    """
    Calculate the integral of the function S(E, E_c) = E/E_c * K_v(E/E_c) from E/E_c to upper_bound,
    where upper bound is some large number (in practice this should be inifinity) and E is energy and E_c is
    critical energy. This function uses vectorize to do this integral for an array of E values.
    """
    # Calculate the ratio of E to E_c
    ratio = E / E_c
    
    # Apply the integrand function over all elements of the array
    # np.vectorize creates a vectorized function (one that can work over entire arrays) for the integrand over the range from 'ratio' to 'upper_bound'
    integrand_array = np.vectorize(integrand)(np.linspace(ratio, upper_bound))
    
    # Calculate the integral of the integrand function using the quad method from scipy
    # quad returns the value of the integral and an estimate of the error (which is ignored with '_')
    integral_result, _ = quad(integrand, ratio, upper_bound)
    
    # Multiply the integral result by the ratio to get the final result
    result = ratio * integral_result
    
    # Return the final result of the integral calculation
    return result

run_no=8
nu = 5/3  # order of the Bessel function
uv_max=124 #eV
uv_min=3.1 #eV
xray_max=124e3 #eV



file_numbers=["00001", "00002", "00003", "00004", "00005", "00006","00007","00008","00009","00010","00011"]
full_energy_array=[]
full_phi_array=[]
full_theta_array=[]
full_synchrotron_array=[]
no_xray_photons=np.empty(0)
no_uv_photons=np.empty(0)
no_other_photons=np.empty(0)
sum_array=np.empty(0)
for file_number in file_numbers:
    filename=f"Data_processing/Parameters/h5files/v3d_synchrotron_{file_number}.h5"

    file=h5py.File(filename, 'r')
    full_energy_array.append(file["Energy"][:])
    full_phi_array.append(file["Phi"][:])
    full_theta_array.append(file["Theta"][:])
    full_synchrotron_array.append(file["Synchrotron3D"][:])
    file.close()

for i in range(0, 11):
    run_no=i
    E_c=full_energy_array[run_no]
    E=full_energy_array[run_no]

    #print("E_c sum:",str(np.sum(e_c)))
    #print("E sum:",str(np.sum(e)))
    #print(e)

    photons_per_crit_energy=full_synchrotron_array[run_no]
    #print("Photons per critical energy sum:",str(np.sum(photons_per_crit_energy)))

    # Initialize an empty list to store the result arrays
    matrix = []

    # Loop through each value of critical energy
    for E_c_val in E_c: #E_c_val is the critical energy from an array E_c of critical energies
        array = []  # Create a new list for each critical energy
        for E_val in E: #creates an array for each energy value
            value = E_val/ E_c_val  # Calculate the normalized value
            array.append(value)  # Append to the array
        matrix.append(array)  # Append the array to the matrix

    # Convert the list of arrays into a 2D NumPy array
    matrix = np.array(matrix)

    matrix_file_name=r'Data_processing\Parameters\matrices\S_matrix_'+str(run_no)+'.npy'

    # # Initialize an empty matrix to store the results of the integral
    start_time = time.time()
    try:
        S_matrix=np.load(matrix_file_name)
    except FileNotFoundError:
        S_matrix = np.zeros_like(matrix)
        # Create an upper bound matrix
        upper_bound_matrix = (E[:, None] / E_c) * 100  # Broadcasting to create a matrix of shape (i, j)

        # Use np.vectorize  to apply the integral to each element of the matrix
        S_matrix = np.vectorize(S_integral2)(E[:, None], E_c, upper_bound_matrix)
        np.save(matrix_file_name, S_matrix)
        S_matrix=np.load(matrix_file_name)



    # End timer
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to compute the matrix: {elapsed_time:.6f} seconds")

    # Perform matrix multiplication
    result_matrix = np.einsum('ij,klj->kli', S_matrix, photons_per_crit_energy)
    normalised_result=result_matrix/np.sum(S_matrix,axis=1)



    photons_per_theta_per_phi=np.sum(full_synchrotron_array[run_no],axis=2)


    phi=full_phi_array[run_no]
    theta=full_theta_array[run_no]*1e3

    photons_per_energy=sum_over_theta_and_phi(normalised_result)

    #integrate to get no. of x-ray photons
    # Mask the data to include only the range of interest
    mask = (E >= uv_max) 
    energies_integration = E[mask]
    photons_integration = photons_per_energy[mask]
    x_ray_photons=integrate.simpson(photons_integration,x=energies_integration)
    no_xray_photons=np.append(no_xray_photons,x_ray_photons)

    #integrate to get no. of UV photons
    # Mask the data to include only the range of interest
    mask = (E >= uv_min) & (E <= uv_max)
    energies_integration = E[mask]
    photons_integration = photons_per_energy[mask]
    uv_photons=integrate.simpson(photons_integration,x=energies_integration)
    no_uv_photons=np.append(no_uv_photons,uv_photons)   

    #integrate to get no. of other photons
    # Mask the data to include only the range of interest
    mask = (E <= uv_min)
    energies_integration = E[mask]
    photons_integration = photons_per_energy[mask]
    other_photons=integrate.simpson(photons_integration,x=energies_integration)
    no_other_photons=np.append(no_other_photons,other_photons)  

    integral_sum=x_ray_photons+uv_photons+other_photons
    sum_array=np.append(sum_array,integral_sum) 


#Plot the integrated number of x-ray photons against distance
distances=[0,1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize=(10, 6))
plt.plot(distances,no_xray_photons,marker='o',color='red',label='X-ray Photons')
plt.plot(distances,no_uv_photons,marker='o',color='blue',label='UV Photons')
plt.plot(distances,no_other_photons,marker='o',color='green',label='Other Photons')
plt.xlabel("Distance (m)")
plt.ylabel("Integrated No. Photons")
plt.title("Integrated No. Photons against Distance")
plt.legend()
plt.savefig("Data_processing/Parameters/plots/carys_synchrotron/integrated_no_photons.png", dpi=300, bbox_inches='tight')
plt.show()

#Plot the number of x-ray photons against distance
non_int_no_xray_photons=np.array([no_xray_photons[0]])
non_int_no_uv_photons=np.array([no_uv_photons[0]])
non_int_no_other_photons=np.array([no_other_photons[0]])
non_int_sum_array=np.array([sum_array[0]])
for i in range (1,11):
    non_int_no_xray_photons=np.append(non_int_no_xray_photons,no_xray_photons[i]-no_xray_photons[i-1])  
    non_int_no_uv_photons=np.append(non_int_no_uv_photons,no_uv_photons[i]-no_uv_photons[i-1])
    non_int_no_other_photons=np.append(non_int_no_other_photons,no_other_photons[i]-no_other_photons[i-1])
    non_int_sum_array=np.append(non_int_sum_array,sum_array[i]-sum_array[i-1])


plt.figure(figsize=(10, 6))
plt.plot(distances,non_int_no_xray_photons, marker='o',color='red',label='X-ray Photons')
plt.plot(distances,non_int_no_uv_photons, marker='o',color='blue',label='UV Photons')
plt.plot(distances,non_int_no_other_photons, marker='o',color='green',label='Other Photons')
plt.xlabel("Distance (m)")
plt.ylabel("No. Photons")
plt.title("No. Photons against Distance")
plt.legend()
plt.savefig("Data_processing/Parameters/plots/carys_synchrotron/no_photons.png", dpi=200, bbox_inches='tight')
plt.show()

percentage_xray_photons=non_int_no_xray_photons/non_int_sum_array
percentage_uv_photons=non_int_no_uv_photons/non_int_sum_array
percentage_other_photons=non_int_no_other_photons/non_int_sum_array


plt.figure(figsize=(10, 6))
plt.plot(distances,percentage_xray_photons, marker='o',color='red',label='X-ray Photons')
plt.plot(distances,percentage_uv_photons, marker='o',color='blue',label='UV Photons')
plt.plot(distances,percentage_other_photons, marker='o',color='green',label='Other Photons')
plt.xlabel("Distance (m)")
plt.ylabel("Percentage Photons")
plt.title("Percentage Photons against Distance")
plt.legend()
plt.savefig("Data_processing/Parameters/plots/carys_synchrotron/percentage_photons.png", dpi=200, bbox_inches='tight')
plt.show()

