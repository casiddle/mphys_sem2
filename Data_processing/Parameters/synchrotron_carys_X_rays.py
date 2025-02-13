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


run_no=8
nu = 5/3  # order of the Bessel function
middle_max=1000 #eV
middle_min=500#eV
xray_max=124e3 #eV
min=124 #eV



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
    e_c=full_energy_array[run_no]
    e=full_energy_array[run_no]

    #print("E_c sum:",str(np.sum(e_c)))
    #print("E sum:",str(np.sum(e)))
    #print(e)

    photons_per_crit_energy=full_synchrotron_array[run_no]
    #print("Photons per critical energy sum:",str(np.sum(photons_per_crit_energy)))

    # Initialize an empty list to store the result arrays
    matrix = []

    # Loop through each value of omega_c
    for ec in e_c:
        array = []  # Create a new list for each omega_c
        for i in e:
            value = i/ ec  # Calculate the normalized value
            array.append(value)  # Append to the array
        matrix.append(array)  # Append the array to the matrix

    # Convert the list of arrays into a 2D NumPy array
    matrix = np.array(matrix)

    matrix_file_name=r'Data_processing\Single_run\matrices\S_matrix_'+str(run_no)+'.npy'

    # # Initialize an empty matrix to store the results of the integral

    try:
        S_matrix=np.load(matrix_file_name)
    except FileNotFoundError:
        S_matrix = np.zeros_like(matrix)

    # Apply the integral to each element in the matrix
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                upper_bound=(e[i]/e_c[j])*100
                S_matrix[i, j] = S_integral(e[i], e_c[j], upper_bound)


        np.save(matrix_file_name, S_matrix)
        S_matrix=np.load(matrix_file_name)


    # Perform matrix multiplication
    #print("S_matrix shape:",S_matrix.shape)
    #print("photons_per_crit_energy shape:",photons_per_crit_energy.shape)
    #result_matrix = np.dot(S_matrix, photons_per_crit_energy)
    result_matrix = np.einsum('ij,klj->kli', S_matrix, photons_per_crit_energy)
    normalised_result=result_matrix/np.sum(S_matrix,axis=1)



    photons_per_theta_per_phi=np.sum(full_synchrotron_array[run_no],axis=2)

    #print("photons_per_theta_per_phi shape:",photons_per_theta_per_phi.shape)
    phi=full_phi_array[run_no]
    theta=full_theta_array[run_no]*1e3

    photons_per_energy=sum_over_theta_and_phi(normalised_result)

    #integrate to get no. of x-ray photons
    # Mask the data to include only the range of interest
    mask = (e >= middle_max) 
    energies_integration = e[mask]
    photons_integration = photons_per_energy[mask]
    x_ray_photons=integrate.simpson(photons_integration,x=energies_integration)
    no_xray_photons=np.append(no_xray_photons,x_ray_photons)

    #integrate to get no. of UV photons
    # Mask the data to include only the range of interest
    mask = (e >= middle_min) & (e <= middle_max)
    energies_integration = e[mask]
    photons_integration = photons_per_energy[mask]
    uv_photons=integrate.simpson(photons_integration,x=energies_integration)
    no_uv_photons=np.append(no_uv_photons,uv_photons)   

    #integrate to get no. of other photons
    # Mask the data to include only the range of interest
    mask =(e >= min) & (e <= middle_min)
    energies_integration = e[mask]
    photons_integration = photons_per_energy[mask]
    other_photons=integrate.simpson(photons_integration,x=energies_integration)
    no_other_photons=np.append(no_other_photons,other_photons)  

    integral_sum=x_ray_photons+uv_photons+other_photons
    sum_array=np.append(sum_array,integral_sum) 


#Plot the integrated number of x-ray photons against distance
distances=[0,1,2,3,4,5,6,7,8,9,10]
plt.figure(figsize=(10, 6))
plt.plot(distances,no_xray_photons,marker='o',color='red',label=f'{middle_max}< X-ray Photons < {xray_max} eV')
plt.plot(distances,no_uv_photons,marker='o',color='blue',label=f'{middle_min}< X-ray Photons < {middle_max} eV')
plt.plot(distances,no_other_photons,marker='o',color='green',label=f'{min}< X-ray Photons < {middle_min} eV')
plt.xlabel("Distance (m)")
plt.ylabel("Integrated No. X-ray Photons")
plt.title("Integrated No. X-ray Photons against Distance")
plt.legend()
plt.savefig("Data_processing/Single_run/plots/carys_synchrotron/x_ray_integrated_no_photons.png", dpi=300, bbox_inches='tight')
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
plt.plot(distances,non_int_no_xray_photons, marker='o',color='red',label=f'{middle_max}< X-ray Photons < {xray_max} eV')    
plt.plot(distances,non_int_no_uv_photons, marker='o',color='blue',label=f'{middle_min}< X-ray Photons < {middle_max} eV')
plt.plot(distances,non_int_no_other_photons, marker='o',color='green',label=f'{min}< X-ray Photons < {middle_min} eV')
plt.xlabel("Distance (m)")
plt.ylabel("No. X-ray Photons")
plt.title("No. X-ray Photons against Distance")
plt.legend()
plt.savefig("Data_processing/Single_run/plots/carys_synchrotron/x_ray_no_photons.png", dpi=200, bbox_inches='tight')
plt.show()

percentage_xray_photons=non_int_no_xray_photons/non_int_sum_array
percentage_uv_photons=non_int_no_uv_photons/non_int_sum_array
percentage_other_photons=non_int_no_other_photons/non_int_sum_array


plt.figure(figsize=(10, 6))
plt.plot(distances,percentage_xray_photons, marker='o',color='red',label=f'{middle_max}< X-ray Photons < {xray_max} eV')
plt.plot(distances,percentage_uv_photons, marker='o',color='blue',label=f'{middle_min}< X-ray Photons < {middle_max} eV')
plt.plot(distances,percentage_other_photons, marker='o',color='green',label=f'{middle_min}< X-ray Photons < {min} eV')
plt.xlabel("Distance (m)")
plt.ylabel("Percentage X-ray Photons")
plt.title("Percentage X-ray Photons against Distance")
plt.legend()
plt.savefig("Data_processing/Single_run/plots/carys_synchrotron/x_ray_percentage_photons.png", dpi=200, bbox_inches='tight')
plt.show()

